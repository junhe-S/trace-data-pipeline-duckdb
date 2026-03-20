"""
WRDS TRACE / FISD Download Script (Python)
Converted from R — requires: wrds, duckdb, pandas, pyarrow, openpyxl, requests

Parallelism: each batch query runs in its own thread (ThreadPoolExecutor).
The WRDS connection is shared across threads (SQLAlchemy pool handles concurrency).
DuckDB writes are serialised with a threading.Lock to avoid corruption.
Tune N_WORKERS to taste — 4 is a safe default; beyond 6 risks WRDS throttling.

Connection safety:
  - _connect_wrds / _disconnect_wrds bracket every public entry point via try/finally.
  - _reconnect_wrds is called automatically when a worker detects a stale connection;
    it retries up to MAX_RECONNECT_ATTEMPTS times with exponential back-off before
    re-raising so the outer try/finally can still clean up.
"""

import io
import logging
import math
import os
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

import duckdb
import pandas as pd
import psycopg2
import requests
import wrds  # pip install wrds

# ── Configuration ─────────────────────────────────────────────────────────────
LOCAL_DBDIR            = "./wrds_trace.duckdb"
N_WORKERS              = 4    # parallel WRDS query threads; increase carefully (max ~6)
MAX_RECONNECT_ATTEMPTS = 3    # how many times _reconnect_wrds will retry
RECONNECT_BACKOFF      = 5    # initial back-off seconds (doubles each attempt)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Column definitions (module-level so callers can import them) ──────────────

COLS_TRACE_STANDARD = [
    "cusip_id", "bond_sym_id", "bsym", "trd_exctn_dt", "trd_exctn_tm",
    "msg_seq_nb", "trc_st", "wis_fl", "cmsn_trd", "ascii_rptd_vol_tx",
    "rptd_pr", "yld_pt", "asof_cd", "side", "diss_rptg_side_cd",
    "orig_msg_seq_nb", "orig_dis_dt", "rptg_party_type", "contra_party_type",
]

COLS_TRACE_ENHANCED = [
    "cusip_id", "bond_sym_id", "trd_exctn_dt", "trd_exctn_tm", "days_to_sttl_ct",
    "lckd_in_ind", "wis_fl", "sale_cndtn_cd", "msg_seq_nb", "trc_st",
    "trd_rpt_dt", "trd_rpt_tm", "entrd_vol_qt", "rptd_pr", "yld_pt",
    "asof_cd", "orig_msg_seq_nb", "rpt_side_cd", "cntra_mp_id",
]


# ── External source registry ──────────────────────────────────────────────────

_EXTERNAL_SOURCES = {
    "liu_wu_yields": (
        "xlsx",
        "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t"
        "/export?format=xlsx&id=11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t",
        {"skiprows": 7, "header": 0},
    ),
    "osbap_linker": (
        "parquet_zip",
        "https://openbondassetpricing.com/wp-content/uploads/2025/11/linker_file_2025.zip",
        {},
    ),
    "siccodes17": (
        "csv_zip",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes17.zip",
        {},
    ),
    "siccodes30": (
        "csv_zip",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes30.zip",
        {},
    ),
    "bbw_4_factors": (
        "xlsx",
        "https://docs.google.com/spreadsheets/d/1stNzNGgu4Varx7_vsTmH1nbo6LiZT8aO"
        "/export?format=xlsx&id=1stNzNGgu4Varx7_vsTmH1nbo6LiZT8aO",
        {},
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# WRDSDownloader — encapsulates connection state and all download logic
# ═════════════════════════════════════════════════════════════════════════════

class WRDSDownloader:
    """
    Manages a single WRDS connection and exposes trace_download,
    fisd_download, and external_data as methods.

    Usage
    -----
        dl = WRDSDownloader(wrds_username="jsmith")
        dl.trace_download("trace_enhanced", "trace_enhanced",
                          column="trd_exctn_dt", columns=COLS_TRACE_ENHANCED)
        dl.external_data()

    The connection is opened lazily the first time a download method is called
    and is always closed in a try/finally block — even if an error occurs
    mid-download.
    """

    def __init__(
        self,
        wrds_username: str = "",
        local_db_path: str = LOCAL_DBDIR,
        n_workers: int = N_WORKERS,
    ) -> None:
        self.wrds_username = wrds_username
        self.local_db_path = local_db_path
        self.n_workers     = n_workers
        self.db: Optional[wrds.Connection] = None
        self._duckdb_lock  = threading.Lock()
        self._reconnect_lock = threading.Lock()   # serialises reconnect attempts
        self.logger        = logging.getLogger(self.__class__.__name__)

    # ── WRDS lifecycle ────────────────────────────────────────────────────────

    def _verify_credentials(self, password: str) -> None:
        """
        Verify username + password with a single raw psycopg2 connection.

        sslmode=require prevents psycopg2 from falling back to a second
        unencrypted attempt (which WRDS blocks), keeping this to exactly
        one TCP connect. On failure we log a clean message and sys.exit()
        immediately — no re-raise, no SQLAlchemy pool noise.
        """
        import sys
        # self.logger.info("Verifying WRDS credentials for '%s' ...", self.wrds_username)
        try:
            conn = psycopg2.connect(
                host="wrds-pgdata.wharton.upenn.edu",
                port=9737,
                dbname="wrds",
                user=self.wrds_username,
                password=password,
                sslmode="require",
                connect_timeout=10,
            )
            conn.close()
            self.logger.info("Credentials verified.")
        except psycopg2.OperationalError as exc:
            self.logger.error(
                "WRDS authentication failed for user '%s'. "
                "Wrong password? Please try again.\n"
                "Error: %s",
                self.wrds_username, exc,
            )
            sys.exit(1)

    def _connect_wrds(self) -> None:
        """
        Connect to WRDS with a guaranteed single-error credential check.

        Flow:
          1. Read password from ~/.pgpass if an entry exists, otherwise
             prompt the user with getpass (input is never echoed).
          2. Verify with a single psycopg2 ping — one TCP connect, no pool,
             so a wrong password raises exactly one error here and stops.
          3. Write the verified password to ~/.pgpass so wrds.Connection()
             can read it silently (no second prompt).
          4. Open wrds.Connection() — succeeds on the first try because
             credentials are already confirmed.
        """
        import getpass

        # ── Step 1: get password ──────────────────────────────────────────────
        pgpass_path = os.path.expanduser("~/.pgpass")
        password: str = ""

        if os.path.exists(pgpass_path):
            with open(pgpass_path) as f:
                for line in f:
                    if "wrds-pgdata" in line:
                        # format: host:port:dbname:user:password
                        parts = line.strip().split(":")
                        if len(parts) == 5:
                            password = parts[4]
                            break

        if not password:
            password = getpass.getpass(
                f"Enter your WRDS password for '{self.wrds_username}': "
            )

        # ── Step 2: verify — one TCP connect, no pool ─────────────────────────
        self._verify_credentials(password)

        # ── Step 3: write verified password to ~/.pgpass ──────────────────────
        entry = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{self.wrds_username}:{password}\n"
        existing = open(pgpass_path).readlines() if os.path.exists(pgpass_path) else []
        lines = [l for l in existing if "wrds-pgdata" not in l] + [entry]
        with open(pgpass_path, "w") as f:
            f.writelines(lines)
        os.chmod(pgpass_path, 0o600)   # postgres requires this

        # ── Step 4: open the pool — credentials confirmed, no prompt ─────────
        self.logger.info("Opening WRDS connection pool ...")
        self.db = wrds.Connection(wrds_username=self.wrds_username)
        self.logger.info("WRDS connection established.")

    def _disconnect_wrds(self) -> None:
        """Close the WRDS connection, swallowing any errors."""
        if self.db is not None:
            try:
                self.db.close()
                self.logger.info("WRDS connection closed.")
            except Exception:
                self.logger.warning("Error while closing WRDS connection (ignored).")
            finally:
                self.db = None

    def _reconnect_wrds(self) -> None:
        """
        Re-establish a broken WRDS connection.

        Called automatically by worker threads when a query fails with a
        connection-level exception.  Uses an exponential back-off and retries
        up to MAX_RECONNECT_ATTEMPTS times.  A module-level lock ensures that
        multiple threads racing to reconnect serialise safely — only the first
        thread actually reconnects; the rest see a healthy connection and
        return immediately.

        Raises the last exception if all attempts are exhausted.
        """
        with self._reconnect_lock:
            # Another thread may have already reconnected while we waited.
            if self.db is not None:
                try:
                    self.db.raw_sql("SELECT 1")
                    self.logger.debug("Reconnect: connection already healthy, skipping.")
                    return
                except Exception:
                    pass   # still broken — fall through to reconnect logic

            last_exc: Optional[Exception] = None
            backoff = RECONNECT_BACKOFF

            for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                try:
                    # Cleanly close whatever is left of the old connection.
                    if self.db is not None:
                        try:
                            self.db.close()
                        except Exception:
                            pass
                        self.db = None

                    self.logger.info(
                        "Reconnecting to WRDS (attempt %d/%d) ...",
                        attempt, MAX_RECONNECT_ATTEMPTS,
                    )
                    self.db = wrds.Connection(wrds_username=self.wrds_username)
                    self.logger.info("WRDS reconnect successful.")
                    return

                except Exception as exc:
                    last_exc = exc
                    self.logger.warning(
                        "WRDS reconnect attempt %d failed: %s. "
                        "Retrying in %ds ...",
                        attempt, exc, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2   # exponential back-off

            self.logger.exception("WRDS reconnect failed after %d attempts.", MAX_RECONNECT_ATTEMPTS)
            raise last_exc  # re-raise so the outer try/finally can clean up

    # ── DuckDB helpers ────────────────────────────────────────────────────────

    def _get_con(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.local_db_path, read_only=read_only)

    @staticmethod
    def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
        result = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        return result[0] > 0

    @staticmethod
    def _build_in_sql(
        schema: str,
        table: str,
        column: str,
        batch: list,
        select_cols: Optional[List[str]] = None,
    ) -> str:
        """
        Build a SQL IN-clause query with values inlined as literals.
        Handles dates, strings, and numeric types correctly.
        Values come from WRDS (trusted source), so inlining is safe.
        NA values are silently dropped from the IN list.
        """
        def is_na(v) -> bool:
            try:
                return v is None or bool(pd.isna(v))
            except (TypeError, ValueError):
                return False

        def fmt(v) -> str:
            if isinstance(v, str):
                return f"'{v}'"
            if hasattr(v, "strftime"):      # Timestamp / datetime.date
                return f"'{v.strftime('%Y-%m-%d')}'"
            return str(v)

        col_clause = ", ".join(f'"{c}"' for c in select_cols) if select_cols else "*"
        clean = [v for v in batch if not is_na(v)]

        if not clean:
            return f"SELECT {col_clause} FROM {schema}.{table} WHERE false"

        literals = ", ".join(fmt(v) for v in clean)
        return (
            f'SELECT {col_clause} FROM {schema}.{table} '
            f'WHERE "{column}" IN ({literals})'
        )

    def _write_chunk(
        self,
        con: duckdb.DuckDBPyConnection,
        local_table: str,
        chunk: pd.DataFrame,
    ) -> None:
        """Insert a DataFrame chunk into DuckDB, creating the table if needed."""
        if self._table_exists(con, local_table):
            con.execute(f'INSERT INTO "{local_table}" SELECT * FROM chunk')
        else:
            con.execute(f'CREATE TABLE "{local_table}" AS SELECT * FROM chunk')

    # ── Worker (runs in a thread) ─────────────────────────────────────────────

    def _fetch_and_store(
        self,
        schema: str,
        wrds_table: str,
        local_table: str,
        batch: list,
        column: str,
        select_cols: Optional[List[str]],
    ) -> int:
        """
        Fetch one batch from WRDS and write to DuckDB.

        If the query raises a connection-level error the worker calls
        _reconnect_wrds() once and retries.  Any other exception propagates
        immediately so ThreadPoolExecutor can surface it to the main thread.

        Returns the number of rows written (0 if the result set is empty).
        """
        sql = self._build_in_sql(schema, wrds_table, column, batch, select_cols)

        # ── Query with one automatic reconnect on connection failure ──────────
        try:
            chunk: pd.DataFrame = self.db.raw_sql(sql)
        except Exception as exc:
            if _is_connection_error(exc):
                self.logger.warning(
                    "Connection error in worker (%s). Attempting reconnect ...", exc
                )
                self._reconnect_wrds()          # raises if all retries exhausted
                chunk = self.db.raw_sql(sql)    # retry the query once
            else:
                raise

        if chunk.empty:
            return 0

        with self._duckdb_lock:
            con = self._get_con()
            self._write_chunk(con, local_table, chunk)
            con.close()

        return len(chunk)

    # ── Generic parallel batched downloader ───────────────────────────────────

    def _batched_download(
        self,
        schema: str,
        wrds_table: str,
        local_table: str,
        frequency: int,
        column: str,
        columns: Optional[List[str]],
        n_workers: int,
    ) -> None:
        """Shared logic for parallel batched incremental downloads."""
        select_cols: Optional[List[str]] = None
        if columns is not None:
            select_cols = (
                list(dict.fromkeys([column] + columns)) if column else list(columns)
            )

        if column:
            # ── Determine which values are still missing locally ──────────────
            con = self._get_con()
            if self._table_exists(con, local_table):
                existing = set(
                    con.execute(f'SELECT DISTINCT "{column}" FROM "{local_table}"')
                    .df()[column]
                    .tolist()
                )
                con.close()
                remote_vals = set(
                    self.db.raw_sql(
                        f'SELECT DISTINCT "{column}" FROM {schema}.{wrds_table}'
                    )[column].tolist()
                )
                values = list(remote_vals - existing)
            else:
                con.close()
                values = (
                    self.db.raw_sql(
                        f'SELECT DISTINCT "{column}" FROM {schema}.{wrds_table}'
                    )[column].tolist()
                )

            if not values:
                self.logger.info("'%s' is already up to date.", local_table)
                return

            # ── Build batch list ──────────────────────────────────────────────
            batches = [
                values[i * frequency : (i + 1) * frequency]
                for i in range(math.ceil(len(values) / frequency))
            ]
            total = len(batches)
            n_workers = min(total, n_workers)
            self.logger.info(
                "Downloading %d batches for '%s' with %d workers ...",
                total, local_table, n_workers,
            )

            # ── Parallel fetch ────────────────────────────────────────────────
            completed = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self._fetch_and_store,
                        schema, wrds_table, local_table,
                        batch, column, select_cols,
                    ): i
                    for i, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    future.result()   # re-raises any worker exception immediately
                    completed += 1
                    self.logger.info(
                        "[%s] Batch %d/%d done (%.1f%%)",
                        local_table, completed, total, completed / total * 100,
                    )

        else:
            # ── Full table download (no batching needed) ──────────────────────
            col_clause = (
                ", ".join(f'"{c}"' for c in columns) if columns else "*"
            )
            sql = f"SELECT {col_clause} FROM {schema}.{wrds_table}"
            chunk = self.db.raw_sql(sql)

            with self._duckdb_lock:
                con = self._get_con()
                if self._table_exists(con, local_table):
                    con.execute(f'DROP TABLE "{local_table}"')
                con.execute(f'CREATE TABLE "{local_table}" AS SELECT * FROM chunk')
                con.close()

            self.logger.info("Table '%s' written (%d rows).", local_table, len(chunk))

    # ── Public download methods ───────────────────────────────────────────────

    def trace_download(
        self,
        wrds_table: str,
        local_table: str,
        frequency: int = 50,
        column: str = "date",
        columns: Optional[List[str]] = None,
        n_workers: Optional[int] = None,
    ) -> None:
        """
        Download (or incrementally update) a TRACE table from WRDS into a local
        DuckDB database, batching by unique values of `column`.

        Requires an active WRDS connection (self.db). Call via run_all(), which
        manages the single connect/disconnect around the full session, or call
        _connect_wrds() / _disconnect_wrds() yourself when using this method
        directly.

        Parameters
        ----------
        wrds_table  : table name inside the 'trace' schema on WRDS
        local_table : target table name in the local DuckDB file
        frequency   : number of column values to pull per batch
        column      : column to batch on; pass '' to download the whole table
        columns     : list of columns to select; None means all columns
        n_workers   : parallel download threads (default: self.n_workers)
        """
        self.logger.info("=== trace_download: %s -> %s ===", wrds_table, local_table)
        self._batched_download(
            "trace", wrds_table, local_table,
            frequency, column, columns,
            n_workers if n_workers is not None else self.n_workers,
        )

    def fisd_download(
        self,
        wrds_table: str,
        local_table: str,
        frequency: int = 50,
        column: str = "trd_exctn_dt",
        columns: Optional[List[str]] = None,
        n_workers: Optional[int] = None,
    ) -> None:
        """
        Download (or incrementally update) a FISD table from WRDS into a local
        DuckDB database, batching by unique values of `column`.

        Requires an active WRDS connection (self.db). Call via run_all(), which
        manages the single connect/disconnect around the full session, or call
        _connect_wrds() / _disconnect_wrds() yourself when using this method
        directly.

        Parameters
        ----------
        wrds_table  : table name inside the 'fisd' schema on WRDS
        local_table : target table name in the local DuckDB file
        frequency   : number of column values to pull per batch
        column      : column to batch on; pass '' to download the whole table
        columns     : list of columns to select; None means all columns
        n_workers   : parallel download threads (default: self.n_workers)
        """
        self.logger.info("=== fisd_download: %s -> %s ===", wrds_table, local_table)
        self._batched_download(
            "fisd", wrds_table, local_table,
            frequency, column, columns,
            n_workers if n_workers is not None else self.n_workers,
        )

    def external_data(self, overwrite: bool = False) -> dict:
        """
        Download external reference datasets and store them in the local DuckDB.

        No WRDS connection is needed here — this method does not call
        _connect_wrds / _disconnect_wrds.  Each dataset is only downloaded if
        its table does not already exist (or if overwrite=True).

        Returns a dict of DataFrames keyed by table name (newly written only).
        """
        self.logger.info("=== external_data ===")
        con = duckdb.connect(self.local_db_path, read_only=False)
        data = {}

        try:
            for tbl_name, (kind, url, kwargs) in _EXTERNAL_SOURCES.items():
                if self._table_exists(con, tbl_name) and not overwrite:
                    self.logger.info(
                        "Skipping '%s' (already exists; pass overwrite=True to refresh).",
                        tbl_name,
                    )
                    continue

                self.logger.info("Downloading '%s' ...", tbl_name)
                df = _fetch_external(kind, url, kwargs)

                if self._table_exists(con, tbl_name):
                    con.execute(f'DROP TABLE "{tbl_name}"')
                con.execute(f'CREATE TABLE "{tbl_name}" AS SELECT * FROM df')
                self.logger.info("Wrote '%s' (%d rows).", tbl_name, len(df))
                data[tbl_name] = df
        finally:
            con.close()

        return data

    def run_all(self) -> None:
        """
        Convenience method: run every TRACE, FISD, and external download in
        sequence under a single WRDS session.

        The connection is opened once here and closed in a finally block,
        so all individual downloads share one login — no repeated prompts.
        external_data() needs no WRDS connection and runs after disconnect.
        """
        self._connect_wrds()   # exits via sys.exit(1) on auth failure — no finally needed
        try:
            # TRACE
            self.trace_download(
                "trace_enhanced", "trace_enhanced",
                column="trd_exctn_dt",
                columns=COLS_TRACE_ENHANCED,
                n_workers=5,
            )
            self.trace_download(
                "trace", "trace",
                frequency=50,
                column="trd_exctn_dt",
                columns=COLS_TRACE_STANDARD,
                n_workers=5,
            )
            self.trace_download(
                "trace_btds144a", "trace_btds144a",
                column="trd_exctn_dt",
                columns=COLS_TRACE_STANDARD,
                n_workers=5,
            )

            # FISD
            self.fisd_download("fisd_mergedissuer",     "fisd_mergedissuer",     column="")
            self.fisd_download("fisd_mergedissue",      "fisd_mergedissue",      column="offering_date")
            self.fisd_download("fisd_amt_out_hist",     "fisd_amt_out_hist",     column="effective_date")
            self.fisd_download("fisd_ratings",          "fisd_ratings",          column="rating_date")
            self.fisd_download("fisd_mergedredemption", "fisd_mergedredemption", column="")

        finally:
            self._disconnect_wrds()   # only reached if _connect_wrds() succeeded

        # External data needs no WRDS connection — runs after disconnect
        self.external_data()


# ── Module-level helpers (no WRDS state needed) ───────────────────────────────

def _is_connection_error(exc: Exception) -> bool:
    """
    Heuristic: return True if the exception looks like a broken/stale
    WRDS/SQLAlchemy connection rather than a query error.

    Covers the most common cases; extend as needed.
    """
    msg = str(exc).lower()
    connection_keywords = (
        "connection", "disconnect", "timeout", "broken pipe",
        "ssl", "eof", "reset by peer", "operationalerror",
    )
    return any(kw in msg for kw in connection_keywords)


def _dl(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _dl_unzip(url: str) -> list:
    """Download a ZIP and extract to a temp directory; return extracted paths."""
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(_dl(url))) as z:
        z.extractall(tmpdir)
    return [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]


def _fetch_external(kind: str, url: str, kwargs: dict) -> pd.DataFrame:
    """Download and parse one external dataset based on its type."""
    if kind == "xlsx":
        return pd.read_excel(io.BytesIO(_dl(url)), **kwargs)
    elif kind == "parquet_zip":
        paths = _dl_unzip(url)
        parquet_path = next(p for p in paths if p.endswith(".parquet"))
        return pd.read_parquet(parquet_path)
    elif kind == "csv_zip":
        paths = _dl_unzip(url)
        csv_path = next(p for p in paths if not p.endswith((".zip", ".pdf")))
        return pd.read_csv(csv_path, sep=r"\s+", header=None, comment="#", on_bad_lines="skip")
    else:
        raise ValueError(f"Unknown external source kind: {kind}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # !! Set your WRDS username here — do not leave it blank.
    # With a username set, wrds will only prompt for your password (once),
    # then save it to ~/.pgpass so future runs need no interaction at all.
    wrds_username = ""
    dl = WRDSDownloader(wrds_username = wrds_username)
    dl.run_all()




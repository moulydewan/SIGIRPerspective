import logging
import ir_datasets
import pandas as pd

logger = logging.getLogger(__name__)

class TRECDLAdapter:
    """Adapter for TREC Deep Learning 2019–2021 datasets (MSMARCO corpus)."""

    def __init__(self, year: int = 2020, mode: str = "passage"):
        if year not in [2019, 2020, 2021]:
            raise ValueError("Supported years are 2019–2021.")
        if mode not in ["passage", "document"]:
            raise ValueError("mode must be 'passage' or 'document'.")

        # Handle dataset naming conventions across years
        if year in [2019, 2020]:
            dataset_id = f"msmarco-{mode}/trec-dl-{year}"
        else:  # 2021
            dataset_id = f"msmarco-{mode}-v2/trec-dl-{year}"

        logger.info(f"Loading {dataset_id} via ir_datasets ...")

        try:
            self.dataset = ir_datasets.load(dataset_id)
        except KeyError:
            raise ValueError(f"Dataset not found in ir_datasets: {dataset_id}")

        self.name = dataset_id
        self.year = year
        self.mode = mode

    def load(self, limit: int = None):
        try:
            docs_iter = self.dataset.docs_iter()
            queries_iter = self.dataset.queries_iter()
            qrels_iter = self.dataset.qrels_iter()

            docs = list(docs_iter if limit is None else [d for _, d in zip(range(limit), docs_iter)])
            queries = list(queries_iter if limit is None else [q for _, q in zip(range(limit), queries_iter)])
            qrels = list(qrels_iter if limit is None else [r for _, r in zip(range(limit), qrels_iter)])

            logger.info(f"Loaded {len(docs)} docs, {len(queries)} queries, {len(qrels)} qrels.")
            return {"docs": docs, "queries": queries, "qrels": qrels}
        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return {"docs": [], "queries": [], "qrels": []}


import logging
import ir_datasets
from typing import Dict, Any, Iterator, Optional

logger = logging.getLogger(__name__)


class TRECDLAdapter:
    """Adapter for TREC Deep Learning 2019–2021 datasets (MSMARCO corpus)."""

    YEARS = [2019, 2020, 2021]
    MODES = ["passage", "document"]

    def __init__(self, year: int = 2020, mode: str = "passage"):
        if year not in self.YEARS:
            raise ValueError(f"Supported years: {self.YEARS}")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of: {self.MODES}")

        # Dataset ID (ir_datasets naming scheme)
        if year in [2019, 2020]:
            dataset_id = f"msmarco-{mode}/trec-dl-{year}"
        else:  # 2021
            dataset_id = f"msmarco-{mode}-v2/trec-dl-{year}"

        logger.info(f"Initializing dataset: {dataset_id}")
        try:
            self.dataset = ir_datasets.load(dataset_id)
        except KeyError:
            raise ValueError(f"Dataset not found in ir_datasets: {dataset_id}")

        self.name = dataset_id
        self.year = year
        self.mode = mode

    # Data loading
    def load(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Load dataset into memory.
        If limit is specified, load that many queries *that actually have judged documents*.
        """
        try:
            # Step 1: Build mapping of query → qrels
            qrel_map = {}
            for r in self.dataset.qrels_iter():
                qrel_map.setdefault(r.query_id, []).append(r)

            # Step 2: Restrict to queries that have qrels
            all_queries = [q for q in self.dataset.queries_iter() if q.query_id in qrel_map]
            if limit:
                all_queries = all_queries[:limit]

            qids = {q.query_id for q in all_queries}

            # Step 3: Gather qrels for these queries
            qrels = [r for r in self.dataset.qrels_iter() if r.query_id in qids]

            # Step 4: Collect all relevant doc IDs
            docids = {r.doc_id for r in qrels}
            docs = [d for d in self.dataset.docs_iter() if d.doc_id in docids]

            logger.info(
                f"[TREC DL {self.year}] Loaded {len(docs)} docs, {len(all_queries)} queries, {len(qrels)} qrels."
            )

            return {"docs": docs, "queries": all_queries, "qrels": qrels}

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return {"docs": [], "queries": [], "qrels": []}


    # Iterators for streaming TRECDL
    def iter_docs(self, limit: Optional[int] = None) -> Iterator[Any]:
        """Yield documents lazily to avoid memory blowup."""
        count = 0
        for doc in self.dataset.docs_iter():
            yield doc
            count += 1
            if limit and count >= limit:
                break

    def iter_queries(self, limit: Optional[int] = None) -> Iterator[Any]:
        """Yield queries."""
        count = 0
        for q in self.dataset.queries_iter():
            yield q
            count += 1
            if limit and count >= limit:
                break

    def iter_qrels(self, limit: Optional[int] = None) -> Iterator[Any]:
        """Yield qrels."""
        count = 0
        for r in self.dataset.qrels_iter():
            yield r
            count += 1
            if limit and count >= limit:
                break

     #helper function to create a dataframe for TREC    
    def trec_df(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert loaded TREC DL data into a flat DataFrame with qid, query, docid, passage, and rel.
        """
        if not all(k in data for k in ["docs", "queries", "qrels"]):
            raise ValueError("Input must contain 'docs', 'queries', and 'qrels' keys.")

        doc_lookup = {d.doc_id: getattr(d, "text", "") for d in data["docs"]}
        query_lookup = {q.query_id: getattr(q, "text", "") for q in data["queries"]}

        rows = []
        for r in data["qrels"]:
            qid = r.query_id
            docid = r.doc_id
            rel = getattr(r, "relevance", 0)
            query = query_lookup.get(qid)
            passage = doc_lookup.get(docid)
            if query and passage:
                rows.append((qid, query, docid, passage, rel))

        df = pd.DataFrame(rows, columns=["qid", "query", "docid", "passage", "rel"])
        return df
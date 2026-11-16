"""Read the frozen Day 6 corpus and reuse its embedder and pgvector search."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from convert_day6 import convert_chunk
from baseline import validate_search

ROOT = Path(__file__).resolve().parents[1]


class LabError(ValueError):
    """An application error whose message contains no connection credentials."""


def error_details(error):
    if isinstance(error, LabError):
        message = str(error)
    elif isinstance(error, FileNotFoundError):
        message = 'Required file missing; check --day6-root and the setup instructions.'
    else:
        message = 'External operation failed; check the database, dependencies and model cache.'
    return {'error_type': type(error).__name__, 'message': message}


def embedding_digest(vectors):
    """Hash little-endian float32 values independently of NPZ container metadata."""
    checksum = hashlib.sha256()
    for vector in vectors:
        checksum.update(struct.pack(f'<{len(vector)}f', *vector))
    return checksum.hexdigest()


def digest(path):
    data = path.read_text(encoding='utf-8').encode('utf-8') if path.suffix == '.py' else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_snapshot(day6_root):
    snapshot = json.loads((ROOT / 'data/day6.json').read_text(encoding='utf-8'))
    for name, expected in snapshot['source_sha256'].items():
        if name in ('results/embedding_report.json', 'results/embeddings.npz') or name.startswith('results/runs/'):
            continue
        if digest(day6_root / name) != expected:
            raise LabError(f'Day 6 snapshot changed: {name}')
    chunks = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
    exported = [convert_chunk(json.loads(line)) for line in
                (day6_root / 'results/chunks_B.jsonl').read_text(encoding='utf-8').splitlines()]
    if chunks != exported or len(chunks) != snapshot['corpus']['chunk_count']:
        raise LabError('Day 7 chunks differ from the frozen Day 6 B export')
    return snapshot, chunks


def verify_index(rows, chunks, vectors, vector_ids, dimensions):
    """Check content and vectors, not just the number of database rows."""
    if (vector_ids != [c['id'] for c in chunks] or len(vectors) != len(chunks)
            or len(rows) != len(chunks) or len({r['chunk_id'] for r in rows}) != len(rows)):
        raise LabError('Database/export/embedding IDs or counts do not match')
    expected = {c['id']: (c, vector) for c, vector in zip(chunks, vectors)}
    for row in rows:
        if row['chunk_id'] not in expected:
            raise LabError('Database contains a different Day 6 snapshot')
        chunk, vector = expected[row['chunk_id']]
        actual = json.loads(row['embedding'])
        if (convert_chunk(row) != chunk or len(vector) != dimensions or len(actual) != dimensions
                or any(not math.isfinite(v) for v in [*vector, *actual])
                or not math.isclose(sum(v * v for v in vector), 1, abs_tol=1e-5)
                or any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(actual, vector))):
            raise LabError(f'Database content or embedding mismatch: {chunk["id"]}')


class Day6Retriever:
    def __init__(self, embedder, store):
        if embedder.dimensions != store.dimensions:
            raise ValueError('Query embedder and vector store dimensions differ')
        self.embedder = embedder
        self.store = store

    def search(self, query, *, filters=None, top_k=3):
        validate_search(query, top_k, filters)
        if filters and set(filters) - {'section'}:
            raise ValueError('Only section filters are supported by the Day 6 store')
        vector = self.embedder.embed_query(query)
        rows = self.store.search('C' if filters else 'B', vector, top_k=top_k, filters=filters)
        return [{**convert_chunk(row), 'score': row['score'], 'rank': rank}
                for rank, row in enumerate(rows, 1)]


def connect_day6(day6_root):
    day6_root = Path(day6_root).resolve()
    snapshot, chunks = load_snapshot(day6_root)
    import numpy as np
    import psycopg
    from dotenv import dotenv_values
    from psycopg.rows import dict_row

    url = os.environ.get('DATABASE_URL') or dotenv_values(ROOT / '.env').get('DATABASE_URL')
    if not url:
        raise LabError('DATABASE_URL is missing; configure the Day 7 .env file')
    url = psycopg.conninfo.make_conninfo(url, options='-c default_transaction_read_only=on')
    settings = snapshot['embedding']
    with np.load(day6_root / 'results/embeddings.npz', allow_pickle=False) as archive:
        vectors, ids = archive['B'].tolist(), archive['B_ids'].tolist()
    if embedding_digest(vectors) != snapshot['embedding_B_sha256']:
        raise LabError('B embeddings changed; restore the frozen Day 6 vectors.')
    with psycopg.connect(url, connect_timeout=5, row_factory=dict_row) as connection:
        connection.read_only = True
        rows = connection.execute('''
            SELECT chunk_id, document_id, source, page, text, metadata, embedding::text
            FROM day6_chunks WHERE strategy = %s ORDER BY chunk_id
        ''', ('B',)).fetchall()
    verify_index(rows, chunks, vectors, ids, settings['dimensions'])

    # A private package name avoids collisions with the Day 7 src directory.
    package = day6_root / 'starter/src/__init__.py'
    spec = importlib.util.spec_from_file_location('_day6', package)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_day6'] = module
    spec.loader.exec_module(module)
    from _day6.embeddings import MyEmbedder
    from _day6.store import PgVectorStore

    embedder = MyEmbedder(settings['model'], settings['revision'])
    if (embedder.dimensions != settings['dimensions']
            or embedder.max_sequence_length != settings['max_sequence_length']
            or not np.allclose(embedder.embed_query(chunks[0]['text']), vectors[0], rtol=1e-5, atol=1e-6)):
        raise LabError('Loaded query model is incompatible with the Day 6 embeddings')
    store = PgVectorStore(url, settings['dimensions'])
    return Day6Retriever(embedder, store), {
        'snapshot_sha256': digest(ROOT / 'data/day6.json'),
        'chunks_sha256': digest(ROOT / 'data/chunks.json'),
        'verified_chunks': len(rows), 'database': store.stats(),
        'embedding': settings, 'reference_embedding_matches': True,
        'embedding_B_sha256': snapshot['embedding_B_sha256'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6-root', required=True, type=Path)
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', default=3, type=int)
    args = parser.parse_args()
    if not args.query.strip() or args.top_k < 1:
        parser.error('query must not be blank and top-k must be positive')
    output = ROOT / 'results/day6' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    phase = 'connect_and_verify'
    started = perf_counter()
    try:
        retriever, verification = connect_day6(args.day6_root)
        setup_seconds = perf_counter() - started
        phase = 'search'
        started = perf_counter()
        results = retriever.search(args.query, top_k=args.top_k)
        trace = {'status': 'ok', 'purpose': 'connection_smoke_test_not_benchmark',
                 'query': args.query, 'filter': None, 'top_k': args.top_k,
                 'setup_seconds': setup_seconds, 'search_seconds': perf_counter() - started,
                 'service_cost_usd': 0, 'verification': verification, 'results': results}
    except Exception as error:
        trace = {'status': 'error', 'phase': phase, **error_details(error)}
    (output / 'trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': trace['status'], 'trace': str(output / 'trace.json')}, ensure_ascii=False))
    if trace['status'] == 'error':
        print(f"{trace['phase']}: {trace['error_type']}: {trace['message']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

import glob
for f in glob.glob('notebooks/*.py') + ['notebooks/01_embeddings_index.ipynb']:
    with open(f, 'rb') as fh:
        data = fh.read()
    crlf = data.count(b'\r\n')
    if crlf > 0:
        print(f'{f}: {crlf} CRLF lines - fixing')
        with open(f, 'wb') as fh:
            fh.write(data.replace(b'\r\n', b'\n'))
    else:
        print(f'{f}: OK (LF)')
"""Check a local archive before uploading it through Dash."""
import sys, zipfile
from pathlib import Path
from collections import Counter
from document_pipeline import ALLOWED, MAX_FILES

def main():
    if len(sys.argv)<2:
        print('Usage: python check_zip.py "C:\\path\\to\\archive.zip"');return 2
    p=Path(sys.argv[1].strip().strip('"'))
    print('Path:',p,'\nExists:',p.is_file())
    if not p.is_file():return 2
    print('Size MB:',round(p.stat().st_size/1048576,2))
    if not zipfile.is_zipfile(p):print('ERROR: Not a valid ZIP');return 2
    with zipfile.ZipFile(p) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        counts=Counter(Path(i.filename).suffix.lower() for i in files)
        print('Files:',len(files),'\nTypes:',dict(counts))
        print('Supported files:',sum(n for ext,n in counts.items() if ext in ALLOWED))
        print('Uncompressed MB:',round(sum(i.file_size for i in files)/1048576,2))
        print('Sample paths:',*[i.filename for i in files[:8]],sep='\n  ')
        if len(files)>MAX_FILES:print('WARNING: Too many files for current extraction limits')
    return 0
if __name__=='__main__':raise SystemExit(main())

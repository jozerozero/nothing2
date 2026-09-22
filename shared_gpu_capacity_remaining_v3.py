"""Read-only remaining user-listed parents; not a borrowing authorization."""
import os,re
import shared_gpu_capacity_probe as original
PARENTS=('206117','206116','194181','208415','208467')
def parent_record(raw,parent,uid=None):
    original.require(parent in PARENTS,'parent outside scope')
    fields=original.parsed_fields(raw)
    uid=os.getuid() if uid is None else uid
    original.require(fields.get('JobId')==parent and fields.get('JobState')=='RUNNING' and fields.get('NumNodes')=='1','parent is not one-node running')
    original.require(re.fullmatch(r'[^()]+\('+str(uid)+r'\)',fields.get('UserId','')) is not None,'wrong owner')
    original.require(re.fullmatch(r'[A-Za-z0-9_-]+',fields.get('NodeList','')) is not None,'invalid node')
    return fields
original.PARENTS=PARENTS
original.parent_record=parent_record
original.__file__=__file__
if __name__=='__main__':raise SystemExit(original.main())

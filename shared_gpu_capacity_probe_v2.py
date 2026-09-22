"""Same read-only evidence collector, expanded only to user-listed 8GPU parents."""
import shared_gpu_capacity_probe as original
PARENTS=('196092','196093','200798','200797','204828','204827','204826','194259','194180')
original.PARENTS=PARENTS
# Child commands and identity receipts refer to this versioned entry, while the
# original collector is separately source pinned by every operational plan.
original.__file__=__file__
if __name__=='__main__':
    raise SystemExit(original.main())

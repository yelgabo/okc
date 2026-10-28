#!/bin/sh
# Per-kernel wgmma pipeline counts from a cuobjdump SASS dump (tools/moe_compile_fused.sh's
# second argument). A pipelined fat kernel shows one `wait1` per wgmma group and a `wait0` only
# per segment end; serialized wgmmas show a `wait0` after every HGMMA.
#   tools/moe_sass_counts.sh moe.sass
awk '/Function :/ { f = $3 }
     /HGMMA/ { h[f]++ }
     /WARPGROUP.DEPBAR.LE gsb0, 0x1/ { w1[f]++ }
     /WARPGROUP.DEPBAR.LE gsb0, 0x0/ { w0[f]++ }
     /WARPGROUP.ARRIVE/ { a[f]++ }
     /HMMA\./ { m[f]++ }
     /LDSM/ { l[f]++ }
     END { for (k in h) printf "%s HGMMA %d wait1 %d wait0 %d arrive %d HMMA %d LDSM %d\n",
                                k, h[k], w1[k], w0[k], a[k], m[k], l[k] }' "$1" \
    | sed -E 's/_ZN[0-9]+_GLOBAL__N__[a-z0-9_]*_[0-9]+//'

"""Owned sparse producer/consumer causal stamps. Generated diagnostic source only."""

def once(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a));return s.replace(a,b)

def patch(s):
    s=once(s,'#include <torch/extension.h>','#include <torch/extension.h>\n#ifndef ONE_WAIT_TRACE\n#define ONE_WAIT_TRACE 0\n#endif')
    s=once(s,'constexpr int STAMP_W = 16;','constexpr int STAMP_W = 56;')
    s=once(s,'    constexpr bool STAMPS = ONE_STAMP && PROD;', '    constexpr bool DETAILS=ONE_WAIT_TRACE && PROD;\n    constexpr bool STAMPS = ONE_STAMP && PROD;')
    s=once(s,'int* const srow = STAMPS && p.dbg != nullptr','int* const srow = (STAMPS || DETAILS) && p.dbg != nullptr')
    start=s.index('    auto srec =');stop=s.index('    long long s_fullw',start)
    segment=s[start:stop];assert segment.count('if constexpr (STAMPS)')==2
    segment=segment.replace('if constexpr (STAMPS)','if constexpr (STAMPS || DETAILS)')
    s=s[:start]+segment+'''    auto detail=[&](int ordinal,int field){
        if constexpr(DETAILS){
            const int slot=ordinal==0?0:ordinal==16?1:ordinal==64?2:ordinal==128?3:-1;
            if(slot>=0)srec(16+slot*10+field,snow());
        }
    };
    auto presence=[&](int ordinal,int field){
        if constexpr(DETAILS){
            const int slot=ordinal==0?0:ordinal==16?1:ordinal==64?2:ordinal==128?3:-1;
            if(slot>=0)srec(16+slot*10+field,ordinal+1);
        }
    };
    int d_cons=0;
'''+s[stop:]
    s=once(s,'    if constexpr (STAMPS) {\n        if (tid == 0) {\n            srec(0, snow());','    if constexpr (STAMPS || DETAILS) {\n        if (tid == 0) {\n            srec(0, snow());')
    s=once(s,'    int s_tiles = 0, s_nseg = 0;','    int s_tiles = 0, s_nseg = 0;\n    if constexpr(DETAILS)if(tid==0)for(int j=0;j<4;++j)srec(16+j*10+1,0);')
    s=once(s,'            long long s_emw = 0;','            long long s_emw = 0;\n            int d_prod=0;\n            if constexpr(DETAILS)if(ptid==0)for(int j=0;j<4;++j)srec(16+j*10,0);')
    begin=s.index('                for (int base = 0; base < ntiles; base += PF)')
    end=s.index('            // Leave only once every stage',begin)
    part=s[begin:end]
    a='                        const long long s_w0 = STAMPS ? clock64() : 0;'
    part=once(part,a,'                        if constexpr(DETAILS)if(ptid==0){presence(d_prod,0);detail(d_prod,2);}\n'+a)
    a='                            s_emw += clock64() - s_w0;\n                        }'
    part=once(part,a,a+'\n                        if constexpr(DETAILS)if(ptid==0)detail(d_prod,3);')
    a='                        cp_async_mbar_arrive_noinc(full0 + 8 * pst);'
    part=once(part,a,a+'\n                        if constexpr(DETAILS)if(ptid==0)detail(d_prod,4);')
    a='                        if (++pst == NS) {'
    part=once(part,a,'                        if constexpr(DETAILS)if(ptid==0){detail(d_prod,5);++d_prod;}\n'+a)
    s=s[:begin]+part+s[end:]
    a='            // PTX exit, not return:'
    s=once(s,a,'            if constexpr(DETAILS)if(ptid==0){srec(2,snow());srec(4,d_prod);}\n'+a)
    a='        auto acquire = [&](int it, int st, bool common = false) {\n            if constexpr (PROD) {'
    s=once(s,a,a+'\n                if constexpr(DETAILS)if(tid==0){detail(d_cons-1,9);presence(d_cons,1);detail(d_cons,6);}')
    a='                mbar_wait(full0 + 8 * st, cph);'
    s=once(s,a,a+'\n                if constexpr(DETAILS)if(tid==0)detail(d_cons,7);')
    a='        auto release = [&](int st) {\n            if constexpr (PROD) {\n                mbar_arrive(empty0 + 8 * st);'
    s=once(s,a,a+'\n                if constexpr(DETAILS)if(tid==0){detail(d_cons,8);++d_cons;}')
    a='        if constexpr (STAMPS && !COOP) {'
    s=once(s,a,'        if constexpr(DETAILS)if(tid==0){srec(3,snow());srec(5,d_cons);}\n'+a)
    s=once(s,'#if ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)', '#if defined(ONE_WAIT_ONLY) && ONE_WAIT_ONLY\n#define ONE_INSTANCES(X) X(1,49) X(8,51)\n#elif ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)')
    return s

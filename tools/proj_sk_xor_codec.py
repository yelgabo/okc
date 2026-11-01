"""Versioned, lossless CUDA-event measurement transport; source is never encoded."""

def encode_report(report):
    import base64,json,struct,lzma
    words=[];copy=dict(report);copy['rows']=[]
    for row in report['rows']:
        r=dict(row);r['samples_us']={}
        for tag,values in row['samples_us'].items():
            offset=len(words)
            for value in values:
                raw=struct.pack('<f',value/1000);ms=struct.unpack('<f',raw)[0]
                assert ms*1000==value, ('non-lossless CUDA event float',value)
                words.append(struct.unpack('<I',raw)[0])
            r['samples_us'][tag]={'offset':offset,'count':len(values)}
        copy['rows'].append(r)
    dictionary=sorted(set(words));assert 0<len(dictionary)<=65536
    ids={word:i for i,word in enumerate(dictionary)};width=1 if len(dictionary)<=256 else 2
    header=json.dumps(copy,separators=(',',':'),sort_keys=True).encode()
    blob=b'SXE2'+struct.pack('<I',len(header))+header+struct.pack('<IIB',len(dictionary),len(words),width)
    blob+=struct.pack('<'+'I'*len(dictionary),*dictionary)
    blob+=struct.pack('<'+('B' if width==1 else 'H')*len(words),*(ids[word] for word in words))
    filters=[dict(id=lzma.FILTER_LZMA2,dict_size=1<<20,lc=3,lp=0,pb=0)]
    packed='SX2:'+base64.b85encode(lzma.compress(blob,format=lzma.FORMAT_XZ,filters=filters)).decode()
    assert len(packed)<9000, ('report transport overflow',len(packed))
    return packed


def decode_report(encoded):
    import base64,json,struct,lzma
    assert encoded.startswith('SX2:')
    blob=lzma.decompress(base64.b85decode(encoded[4:]),memlimit=16<<20)
    assert len(blob)>=8 and blob[:4]==b'SXE2'
    n=struct.unpack_from('<I',blob,4)[0];start=8+n
    assert 0<n and start+9<=len(blob)
    result=json.loads(blob[8:start]);unique,count,width=struct.unpack_from('<IIB',blob,start);start+=9
    assert 0<unique<=count<=10800 and width==(1 if unique<=256 else 2)
    assert len(blob)==start+unique*4+count*width
    dictionary=struct.unpack_from('<'+'I'*unique,blob,start);start+=unique*4
    assert tuple(sorted(set(dictionary)))==dictionary
    ids=struct.unpack_from('<'+('B' if width==1 else 'H')*count,blob,start)
    assert all(i<unique for i in ids)
    floats=[struct.unpack('<f',struct.pack('<I',dictionary[i]))[0] for i in ids]
    used=[]
    for row in result['rows']:
        for tag,ref in row['samples_us'].items():
            a=ref['offset'];length=ref['count'];b=a+length
            assert type(a) is int and type(length) is int and 0<=a<b<=len(floats)
            assert length==result['rounds']==24
            row['samples_us'][tag]=[1000*value for value in floats[a:b]]
            used.extend(range(a,b))
    assert sorted(used)==list(range(len(floats))), 'missing/overlapping sample slices'
    assert result['format']=='PROJ_SK_XOR_V1'
    assert len(result['rows'])==90 and len(result['seeds'])==3
    assert len({(r['layer'],r['m'],r['seed']) for r in result['rows']})==90
    return result

| Load | Projection | Public v18 µs | C3 range µs | C3 median minus v18 µs |
|---|---|---:|---:|---:|
| decode 1 | qkv_gate | 9.341 | 8.829–9.011 | -0.461 |
| decode 1 | o | 5.101 | 5.197–5.429 | +0.159 |
| decode 1 | pooler | 13.293 | 13.227–13.584 | +0.194 |
| decode 1 | unpooler | 3.722 | 3.107–3.806 | -0.209 |
| decode 1 | router | 3.942 | 3.688–3.885 | -0.235 |
| decode 16 | qkv_gate | 10.255 | 9.518–10.142 | -0.168 |
| decode 16 | o | 5.476 | 5.657–5.904 | +0.297 |
| decode 16 | pooler | 13.984 | 14.400–14.749 | +0.529 |
| decode 16 | unpooler | 13.740 | 13.671–14.102 | +0.219 |
| decode 16 | router | 4.230 | 4.382–4.860 | +0.322 |
| decode 32 | qkv_gate | 12.162 | 10.591–11.007 | -1.173 |
| decode 32 | o | 5.807 | 6.030–6.217 | +0.272 |
| decode 32 | pooler | 16.134 | 14.948–15.065 | -1.115 |
| decode 32 | unpooler | 15.605 | 14.485–14.969 | -1.108 |
| decode 32 | router | 5.043 | 5.002–5.376 | +0.298 |
| verify 1 | qkv_gate | 9.693 | 9.126–10.348 | +0.098 |
| verify 1 | o | 5.496 | 5.074–5.242 | -0.342 |
| verify 1 | pooler | 13.608 | 13.358–13.476 | -0.163 |
| verify 1 | unpooler | 4.282 | 4.205–4.580 | +0.192 |
| verify 1 | router | 4.343 | 4.150–4.433 | -0.171 |
| verify 16 | qkv_gate | 13.987 | 14.460–15.133 | +1.005 |
| verify 16 | o | 6.888 | 7.135–7.432 | +0.368 |
| verify 16 | pooler | 20.489 | 19.702–20.689 | -0.455 |
| verify 16 | unpooler | 19.374 | 18.147–18.918 | -0.583 |
| verify 16 | router | 6.195 | 6.355–6.874 | +0.447 |
| verify 32 | qkv_gate | 17.092 | 17.471–17.771 | +0.428 |
| verify 32 | o | 8.979 | 8.660–9.355 | -0.077 |
| verify 32 | pooler | 24.192 | 23.875–24.517 | +0.182 |
| verify 32 | unpooler | 23.317 | 23.184–23.621 | +0.190 |
| verify 32 | router | 8.441 | 8.139–8.451 | -0.168 |

- 61330d87d32c4190 retrospective factor 1.018023
- 7a15f46e673b43d5 retrospective factor 1.016655
- 2383f5173a934c63 retrospective factor 1.016060

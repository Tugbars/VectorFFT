# The FP32 ZTURN-T ceiling at 2^20 (experiment, not shipped): the 2^20 races on a scratch store

```
recal20_ip0.txt:[k1fs] N=1048576: 5 split(s); row cells borrowed where banked (VFFT_HIGHN_PLAN=borrow)
recal20_ip0.txt:[k1plan] N=1048576: IL plan race (solos, pairs x forms, ZTURN-T chains x widths, chain3 x forms, bwd forms) — a cold cell takes seconds
recal20_ip0.txt:[k1plan] N=1048576: ilp=1399912ns -> ILP (route 9, 0.0, 1 line(s) banked, ord=nat cell)
recal20_ip0.txt:f32 N=1048576 nat oop T=1 recalibrate=0: banked route=ztt (43723 ms)
recal20_ip1.txt:[k1fs] N=1048576: 5 split(s); row cells borrowed where banked (VFFT_HIGHN_PLAN=borrow)
recal20_ip1.txt:[k1plan] N=1048576: IL plan race (solos, pairs x forms, ZTURN-T chains x widths, chain3 x forms, bwd forms) — a cold cell takes seconds
recal20_ip1.txt:[k1plan] N=1048576: ilp=1535950ns -> ILP (route 9, 0.0, 1 line(s) banked, ord=nat cell)
recal20_ip1.txt:f32 N=1048576 nat ip T=1 recalibrate=0: banked route=ztt (46508 ms)
```

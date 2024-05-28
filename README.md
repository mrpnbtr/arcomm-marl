# arcomm-marl
Codes used for ARCOMM-MARL. 
<br>Environments from [SMAC](https://github.com/oxwhirl/smac) and [MA-gym](https://github.com/koulanurag/ma-gym).

Some examples on how to run:
* Run with QMIX+ARCOMM (without and with compression)
```
python main.py --env 3s_vs_5z --n_steps 1500000 --alg qmix --cuda True --arcomm True
python main.py --env 2c_vs_64zg --n_steps 1500000 --alg qmix --cuda True --arcomm True --msg_cut 0.4
```
* Run with normal QMIX or with VDN/QMIX+COMMNET
```
python main.py --env 3s_vs_5z --n_steps 1500000 --alg qmix --cuda True
python main.py --env 2c_vs_64zg --alg <vdn,qmix>+commnet --n_steps 2000000 --cuda True
```

Codes for TARMAC were used from [this repository](https://github.com/TonghanWang/NDQ).

**iperf**

`iperf3 -s -p 5001`

`iperf3 -c <IP> -u -b <bandwidth> -l <packet size> -t <time>`

**tc**

```
sudo tc qdisc add dev eth root tbf \
  rate 10mbit        \  
  burst 32kbit       \ 
  latency 400ms         
```


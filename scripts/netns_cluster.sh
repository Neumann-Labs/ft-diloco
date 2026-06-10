#!/usr/bin/env bash
# Per-replica network namespaces on worker4 so each replica's sync traffic crosses
# a measurable veth (loopback traffic between same-host procs is invisible to NIC
# counters). Also the M4 attachment point for netem WAN profiles.
#
#   sudo scripts/netns_cluster.sh up <N>       create ns ftd0..ftd<N-1> + bridge
#   sudo scripts/netns_cluster.sh down <N>     tear down
#   sudo scripts/netns_cluster.sh netem <i> <netem args...>   e.g. rate 50mbit delay 20ms loss 1%
#   sudo scripts/netns_cluster.sh clear-netem <i>
#
# Topology: br-ftd (10.77.0.1/24, host) <-veth-> ftd<i> (10.77.0.1<i>/24).
# Namespaces reach the LAN (worker1 lighthouse) via host NAT; replica<->replica
# traffic stays on the bridge. Counters: ip -s link show vftd<i> (host side).
set -euo pipefail

BR=br-ftd
SUBNET=10.77.0
CMD="${1:?up|down|netem|clear-netem}"

up() {
  local n="${1:?num namespaces}"
  ip link add "$BR" type bridge 2>/dev/null || true
  ip addr add "$SUBNET.1/24" dev "$BR" 2>/dev/null || true
  ip link set "$BR" up
  for i in $(seq 0 $((n - 1))); do
    ip netns add "ftd$i"
    ip link add "vftd$i" type veth peer name eth0 netns "ftd$i"
    ip link set "vftd$i" master "$BR"
    ip link set "vftd$i" up
    ip netns exec "ftd$i" ip addr add "$SUBNET.1$i/24" dev eth0
    ip netns exec "ftd$i" ip link set eth0 up
    ip netns exec "ftd$i" ip link set lo up
    ip netns exec "ftd$i" ip route add default via "$SUBNET.1"
  done
  sysctl -qw net.ipv4.ip_forward=1
  iptables -t nat -C POSTROUTING -s "$SUBNET.0/24" ! -d "$SUBNET.0/24" -j MASQUERADE 2>/dev/null ||
    iptables -t nat -A POSTROUTING -s "$SUBNET.0/24" ! -d "$SUBNET.0/24" -j MASQUERADE
  # Docker sets FORWARD policy DROP (and loads br_netfilter, so even bridged
  # ns<->ns traffic hits iptables) — accept our subnet explicitly.
  iptables -C FORWARD -s "$SUBNET.0/24" -j ACCEPT 2>/dev/null ||
    iptables -I FORWARD 1 -s "$SUBNET.0/24" -j ACCEPT
  iptables -C FORWARD -d "$SUBNET.0/24" -j ACCEPT 2>/dev/null ||
    iptables -I FORWARD 1 -d "$SUBNET.0/24" -j ACCEPT
  echo "up: $n namespaces on $BR"
}

down() {
  local n="${1:?num namespaces}"
  for i in $(seq 0 $((n - 1))); do
    ip netns del "ftd$i" 2>/dev/null || true
  done
  iptables -t nat -D POSTROUTING -s "$SUBNET.0/24" ! -d "$SUBNET.0/24" -j MASQUERADE 2>/dev/null || true
  iptables -D FORWARD -s "$SUBNET.0/24" -j ACCEPT 2>/dev/null || true
  iptables -D FORWARD -d "$SUBNET.0/24" -j ACCEPT 2>/dev/null || true
  ip link del "$BR" 2>/dev/null || true
  echo "down"
}

netem() {
  local i="${1:?ns index}"
  shift
  # shaping on the host-side veth = the replica's "WAN uplink"
  tc qdisc replace dev "vftd$i" root netem "$@"
  ip netns exec "ftd$i" tc qdisc replace dev eth0 root netem "$@"
  echo "netem ftd$i: $*"
}

clear_netem() {
  local i="${1:?ns index}"
  tc qdisc del dev "vftd$i" root 2>/dev/null || true
  ip netns exec "ftd$i" tc qdisc del dev eth0 root 2>/dev/null || true
  echo "netem cleared ftd$i"
}

case "$CMD" in
  up) up "${2:?}" ;;
  down) down "${2:?}" ;;
  netem) shift; netem "$@" ;;
  clear-netem) clear_netem "${2:?}" ;;
  *) echo "unknown command $CMD" >&2; exit 1 ;;
esac

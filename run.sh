# shellcheck disable=SC2006
py_path=`which python`
run() {
    number=$1
    shift
    for i in $(seq $number); do
      # shellcheck disable=SC2068
      $@
      $py_path main.py  --config 'config/wos11967/wos_init_1500_batch_100_freq_1500.json'
    done
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run "$1"
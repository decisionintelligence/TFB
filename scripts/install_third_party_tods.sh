#!/usr/bin/env bash

rm -rf   ../ts_benchmark/baselines/third_party/tods

REPO_URL='https://gitee.com/zhangbuang/tods.git'

# Clone repository to specified directory
git clone $REPO_URL

mv  ./tods/tods ../ts_benchmark/baselines/third_party
rm -rf ./tods
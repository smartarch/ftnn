#!/bin/bash

set -e

hostType=centos7-minimal

SCRIPT_PATH=$(dirname $(realpath -s $0))
. $SCRIPT_PATH/functions

performInstallLocal "$#" false
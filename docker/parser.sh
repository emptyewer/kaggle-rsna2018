#!/bin/bash
set -e

# The functions in this script parse = separated key value pairs in
# command line arguments like so, --key=value
#
# For illustration purpose take the following command line example
#
# bash script.sh hello --param=value --force --multi-value=1 --multi-value=2 world
#
# Within script.sh add the following lines:
#
# 1. source parse.sh
# 2. eval $(parse_params "$@")
#
# the parser "parse_params" will generate the following variables
#
# $param == value
# $force == "force"
# ${multi-value[0]} == 1
# ${multi-value[1]} == 2
# ${ARGV[0]} == hello
# ${ARGV[1]} == world
#
#
# NOTE: "parse_equal_delimited_params" function will only parse "--key=value" command line
# arguments and positional arguments but will ignore toggle params like "--force" in the
# above example.

function parse_params ()
{
    local existing_named
    local ARGV=()
    echo "local ARGV=(); "
    while [[ "$1" != "" ]]; do
        # If equals delimited named parameter
        if [[ "$1" =~ ^..*=..* ]]; then
            # key is part before first =
            local _key=$(echo "$1" | cut -d = -f 1)
            # val is everything after key and = (protect from param==value error)
            local _val="${1/$_key=}"
            # remove dashes from key name
            _key=${_key//\-}
            # search for existing parameter name
            if (echo "$existing_named" | grep "\b$_key\b" >/dev/null); then
                # if name already exists then it's a multi-value named parameter
                # re-declare it as an array if needed
                if ! (declare -p _key 2> /dev/null | grep -q 'declare \-a'); then
                    echo "$_key=(\"\$$_key\");"
                fi
                # append new value
                echo "$_key+=('$_val');"
            else
                # single-value named parameter
                echo "local $_key=\"$_val\";"
                existing_named=" $_key"
            fi
        # If standalone named parameter
        elif [[ "$1" =~ ^\-. ]]; then
            # remove dashes
            local _key=${1//\-}
            echo "local $_key=\"$_key\";"
        # non-named parameter
        else
            # escape asterisk to prevent bash asterisk expansion
            _escaped=${1/\*/\'\"*\"\'}
            echo "ARGV+=('$_escaped');"
        fi
        shift
    done
}

function parse_equal_delimited_params() {
    local existing_named
    local ARGV=()
    while [[ "$1" != "" ]]; do
        if [[ "$1" =~ ^..*=..* ]]; then
            # key is part before first =
            local _key=$(echo "$1" | cut -d = -f 1)
            # val is everything after key and = (protect from param==value error)
            local _val="${1/$_key=}"
            # remove dashes from key name
            _key=${_key//\-}
            # search for existing parameter name
            if (echo "$existing_named" | grep "\b$_key\b" >/dev/null); then
                # if name already exists then it's a multi-value named parameter
                # re-declare it as an array if needed
                if ! (declare -p _key 2> /dev/null | grep -q 'declare \-a'); then
                    echo "$_key=(\"\$$_key\");"
                fi
                # append new value
                echo "$_key+=('$_val');"
            else
                # single-value named parameter
                echo "local $_key=\"$_val\";"
                existing_named=" $_key"
            fi
        else
            _escaped=${1/\*/\'\"*\"\'}
            echo "ARGV+=('$_escaped');"
        fi
        shift
    done
}

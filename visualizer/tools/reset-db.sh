#!/bin/sh

sudo mysql -e 'drop database trustvis; create database trustvis; connect trustvis; \. tools/trustvis-XXXX-XX-XX-dump.sql'

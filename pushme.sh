#!/bin/bash
msg=${1:-"Update project"}
git add -A
git commit -m "$msg"
git push

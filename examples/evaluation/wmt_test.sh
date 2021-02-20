#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASE_DIR=$(pwd)

OUTPUT_DIR=${OUTPUT_DIR:-$BASE_DIR/wmt14/de_en}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

mkdir -p $OUTPUT_DIR_DATA


echo "Downloading test sets"
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz https://www.statmt.org/wmt14/test-filtered.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Convert SGM files
# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/test/newstest2014.src.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/newstest2014.ref.en
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-src.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/newstest2014.src.en
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-ref.de.sgm \
  > ${OUTPUT_DIR_DATA}/test/newstest2014.ref.de


# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/test/newstest20*.de ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/newstest20*.en ${OUTPUT_DIR}

rm -rf ${OUTPUT_DIR_DATA}/test
rm -f ${OUTPUT_DIR_DATA}/test.tgz
echo "All done."
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

# Clone Moses
  if [ ! -d "${BASE_DIR}/mosesdecoder" ]; then
    echo "Cloning moses for data processing"
    git clone https://github.com/moses-smt/mosesdecoder.git "${BASE_DIR}/mosesdecoder"
  fi





#
#-------------------------------------------------------WMT 2014-----------------------------------------------------
#
OUTPUT_DIR=${BASE_DIR}"/wmt14"

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

mkdir -p $OUTPUT_DIR_DATA


  echo "Downloading test sets"
  wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz https://www.statmt.org/wmt14/test-filtered.tgz

  if test -f "${OUTPUT_DIR_DATA}/test.tgz"; then
    echo "${OUTPUT_DIR_DATA}/test.tgz downloaded."
  else
    echo "Not found URL https://www.statmt.org/wmt14/test.tgz."
    exit
  fi

  # Extract everything
  echo "Extracting all files..."
  mkdir -p "${OUTPUT_DIR_DATA}/test"
  tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"


  mkdir ${OUTPUT_DIR_DATA}/de_en
  mkdir ${OUTPUT_DIR_DATA}/fr_en
  mkdir ${OUTPUT_DIR_DATA}/ru_en
  mkdir ${OUTPUT_DIR_DATA}/de_en/test
  mkdir ${OUTPUT_DIR_DATA}/fr_en/test
  mkdir ${OUTPUT_DIR_DATA}/ru_en/test
  mkdir ${OUTPUT_DIR}/de_en
  mkdir ${OUTPUT_DIR}/fr_en
  mkdir ${OUTPUT_DIR}/ru_en

  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-src.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2014.src.de
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2014.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2014.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-deen-ref.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2014.ref.de


  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.de ${OUTPUT_DIR}/de_en
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.en ${OUTPUT_DIR}/de_en


  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-fren-src.fr.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2014.src.fr
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-fren-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2014.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-fren-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2014.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-fren-ref.fr.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2014.ref.fr


  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/fr_en/test/newstest20*.fr ${OUTPUT_DIR}/fr_en
  cp ${OUTPUT_DIR_DATA}/fr_en/test/newstest20*.en ${OUTPUT_DIR}/fr_en


  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-ruen-src.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2014.src.ru
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-ruen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2014.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-ruen-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2014.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2014-ruen-ref.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2014.ref.ru


  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.ru ${OUTPUT_DIR}/ru_en
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.en ${OUTPUT_DIR}/ru_en

  rm -rf ${OUTPUT_DIR_DATA}/test
  rm -f ${OUTPUT_DIR_DATA}/test.tgz








#
#-------------------------------------------------------WMT 2015-----------------------------------------------------
#
  OUTPUT_DIR=${BASE_DIR}"/wmt15"

  OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

  mkdir -p $OUTPUT_DIR_DATA


  echo "Downloading test sets"
  wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz https://www.statmt.org/wmt15/test.tgz
  size=$(stat --printf="%s" ${OUTPUT_DIR_DATA}/test.tgz)
  if [ $size -eq 0 ]; then
    rm -f ${OUTPUT_DIR_DATA}/test.tgz
  fi

  if test -f "${OUTPUT_DIR_DATA}/test.tgz"; then
    echo "${OUTPUT_DIR_DATA}/test.tgz downloaded."
  else
    echo "Not found URL https://www.statmt.org/wmt15/test.tgz."
    exit
  fi

  # Extract everything
  echo "Extracting all files..."
  mkdir -p "${OUTPUT_DIR_DATA}/test"
  tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"


  mkdir ${OUTPUT_DIR_DATA}/de_en
  mkdir ${OUTPUT_DIR_DATA}/fr_en
  mkdir ${OUTPUT_DIR_DATA}/ru_en
  mkdir ${OUTPUT_DIR_DATA}/de_en/test
  mkdir ${OUTPUT_DIR_DATA}/fr_en/test
  mkdir ${OUTPUT_DIR_DATA}/ru_en/test
  mkdir ${OUTPUT_DIR}/de_en
  mkdir ${OUTPUT_DIR}/fr_en
  mkdir ${OUTPUT_DIR}/ru_en

  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-deen-src.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2015.src.de
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-deen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2015.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-ende-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2015.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-ende-ref.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2015.ref.de

  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.de ${OUTPUT_DIR}/de_en
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.en ${OUTPUT_DIR}/de_en


  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newsdiscusstest2015-fren-src.fr.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2015.src.fr
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newsdiscusstest2015-fren-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2015.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newsdiscusstest2015-enfr-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2015.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newsdiscusstest2015-enfr-ref.fr.sgm \
    > ${OUTPUT_DIR_DATA}/fr_en/test/newstest2015.ref.fr


  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/fr_en/test/newstest20*.fr ${OUTPUT_DIR}/fr_en
  cp ${OUTPUT_DIR_DATA}/fr_en/test/newstest20*.en ${OUTPUT_DIR}/fr_en


  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-ruen-src.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2015.src.ru
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-ruen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2015.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-enru-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2015.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2015-enru-ref.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2015.ref.ru


  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.ru ${OUTPUT_DIR}/ru_en
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.en ${OUTPUT_DIR}/ru_en

  rm -rf ${OUTPUT_DIR_DATA}/test
  rm -f ${OUTPUT_DIR_DATA}/test.tgz







#
#-------------------------------------------------------WMT 2016-----------------------------------------------------
#
  OUTPUT_DIR=${BASE_DIR}"/wmt16"

  OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

  mkdir -p $OUTPUT_DIR_DATA


  echo "Downloading test sets"
  wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz http://data.statmt.org/wmt16/translation-task/test.tgz
  size=$(stat --printf="%s" ${OUTPUT_DIR_DATA}/test.tgz)
  if [ $size -eq 0 ]; then
    rm -f ${OUTPUT_DIR_DATA}/test.tgz
  fi

  if test -f "${OUTPUT_DIR_DATA}/test.tgz"; then
    echo "${OUTPUT_DIR_DATA}/test.tgz downloaded."
  else
    echo "Not found URL http://data.statmt.org/wmt16/translation-task/test.tgz."
    exit
  fi

  # Extract everything
  echo "Extracting all files..."
  mkdir -p "${OUTPUT_DIR_DATA}/test"
  tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"


  mkdir ${OUTPUT_DIR_DATA}/de_en
  mkdir ${OUTPUT_DIR_DATA}/ru_en
  mkdir ${OUTPUT_DIR_DATA}/de_en/test
  mkdir ${OUTPUT_DIR_DATA}/ru_en/test
  mkdir ${OUTPUT_DIR}/de_en
  mkdir ${OUTPUT_DIR}/ru_en

  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-deen-src.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2016.src.de
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-deen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2016.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-ende-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2016.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-ende-ref.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2016.ref.de

  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.de ${OUTPUT_DIR}/de_en
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.en ${OUTPUT_DIR}/de_en


  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-ruen-src.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2016.src.ru
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-ruen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2016.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-enru-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2016.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2016-enru-ref.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2016.ref.ru

  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.ru ${OUTPUT_DIR}/ru_en
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.en ${OUTPUT_DIR}/ru_en

  rm -rf ${OUTPUT_DIR_DATA}/test
  rm -f ${OUTPUT_DIR_DATA}/test.tgz









#
#-------------------------------------------------------WMT 2017-----------------------------------------------------
#
  OUTPUT_DIR=${BASE_DIR}"/wmt17"

  OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

  mkdir -p $OUTPUT_DIR_DATA


  echo "Downloading test sets"
  wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz http://data.statmt.org/wmt17/translation-task/test.tgz
  size=$(stat --printf="%s" ${OUTPUT_DIR_DATA}/test.tgz)
  if [ $size -eq 0 ]; then
    rm -f ${OUTPUT_DIR_DATA}/test.tgz
  fi

  if test -f "${OUTPUT_DIR_DATA}/test.tgz"; then
    echo "${OUTPUT_DIR_DATA}/test.tgz downloaded."
  else
    echo "Not found URL http://data.statmt.org/wmt17/translation-task/test.tgz."
    exit
  fi

  # Extract everything
  echo "Extracting all files..."
  mkdir -p "${OUTPUT_DIR_DATA}/test"
  tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"


  mkdir ${OUTPUT_DIR_DATA}/de_en
  mkdir ${OUTPUT_DIR_DATA}/ru_en
  mkdir ${OUTPUT_DIR_DATA}/zh_en
  mkdir ${OUTPUT_DIR_DATA}/de_en/test
  mkdir ${OUTPUT_DIR_DATA}/ru_en/test
  mkdir ${OUTPUT_DIR_DATA}/zh_en/test
  mkdir ${OUTPUT_DIR}/de_en
  mkdir ${OUTPUT_DIR}/ru_en
  mkdir ${OUTPUT_DIR}/zh_en

  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-deen-src.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2017.src.de
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-deen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2017.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-ende-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2017.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-ende-ref.de.sgm \
    > ${OUTPUT_DIR_DATA}/de_en/test/newstest2017.ref.de

  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.de ${OUTPUT_DIR}/de_en
  cp ${OUTPUT_DIR_DATA}/de_en/test/newstest20*.en ${OUTPUT_DIR}/de_en


  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-ruen-src.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2017.src.ru
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-ruen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2017.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-enru-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2017.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-enru-ref.ru.sgm \
    > ${OUTPUT_DIR_DATA}/ru_en/test/newstest2017.ref.ru

  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.ru ${OUTPUT_DIR}/ru_en
  cp ${OUTPUT_DIR_DATA}/ru_en/test/newstest20*.en ${OUTPUT_DIR}/ru_en


  # Convert SGM files
  # Convert newstest2014 data into raw text format
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-zhen-src.zh.sgm \
    > ${OUTPUT_DIR_DATA}/zh_en/test/newstest2017.src.zh
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-zhen-ref.en.sgm \
    > ${OUTPUT_DIR_DATA}/zh_en/test/newstest2017.ref.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-enzh-src.en.sgm \
    > ${OUTPUT_DIR_DATA}/zh_en/test/newstest2017.src.en
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
    < ${OUTPUT_DIR_DATA}/test/test/newstest2017-enzh-ref.zh.sgm \
    > ${OUTPUT_DIR_DATA}/zh_en/test/newstest2017.ref.zh

  # Copy dev/test data to output dir
  cp ${OUTPUT_DIR_DATA}/zh_en/test/newstest20*.zh ${OUTPUT_DIR}/zh_en
  cp ${OUTPUT_DIR_DATA}/zh_en/test/newstest20*.en ${OUTPUT_DIR}/zh_en

  rm -rf ${OUTPUT_DIR_DATA}/test
  rm -f ${OUTPUT_DIR_DATA}/test.tgz


rm -r\f "${BASE_DIR}/mosesdecoder"
echo "All done."

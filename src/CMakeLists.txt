# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.!


set(ABSL_FLAGS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/absl/flags/flag.cc)


if (SPM_USE_BUILTIN_PROTOBUF)
  set(SPM_PROTO_HDRS builtin_pb/discretepiece.pb.h)
  set(SPM_PROTO_SRCS builtin_pb/discretepiece.pb.cc)
  set(SPM_MODEL_PROTO_HDRS builtin_pb/discretepiece_model.pb.h)
  set(SPM_MODEL_PROTO_SRCS builtin_pb/discretepiece_model.pb.cc)
  set(PROTOBUF_LITE_LIBRARY "")
  set(PROTOBUF_LITE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/arena.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/arenastring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/bytestream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/coded_stream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/extension_set.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/generated_enum_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/generated_message_table_driven_lite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/generated_message_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/implicit_weak_message.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/int128.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/io_win32.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/message_lite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/parse_context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/repeated_field.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/status.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/statusor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/stringpiece.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/stringprintf.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/structurally_valid.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/strutil.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/time.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/wire_format_lite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/zero_copy_stream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/zero_copy_stream_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/zero_copy_stream_impl_lite.cc)
  if (MSVC)
    add_definitions("/DHAVE_PTHREAD /wd4018 /wd4514")
  else()
    add_definitions("-pthread -DHAVE_PTHREAD=1 -Wno-sign-compare -Wno-deprecated-declarations")
  endif()
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite)
  include_directories(builtin_pb)
else()
  find_package(Protobuf REQUIRED)
  include_directories(${Protobuf_INCLUDE_DIRS})
  protobuf_generate_cpp(SPM_PROTO_SRCS SPM_PROTO_HDRS sentencepiece.proto)
  protobuf_generate_cpp(SPM_MODEL_PROTO_SRCS SPM_MODEL_PROTO_HDRS sentencepiece_model.proto)
  set(PROTOBUF_LITE_SRCS "")
  include_directories(${PROTOBUF_INCLUDE_DIR})
  if (MSVC)
    add_definitions("/D_USE_EXTERNAL_PROTOBUF")
  else()
    add_definitions("-D_USE_EXTERNAL_PROTOBUF")
  endif()
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

if (MSVC)
  add_definitions("/D_USE_INTERNAL_STRING_VIEW")
else()
  add_definitions("-D_USE_INTERNAL_STRING_VIEW")
endif()

set(SPM_TRAIN_SRCS
  ${PROTOBUF_LITE_SRCS}
  ${SPM_PROTO_HDRS}
  ${SPM_PROTO_SRCS}
  ${SPM_MODEL_PROTO_HDRS}
  ${SPM_MODEL_PROTO_SRCS}
  ${ABSL_STRINGS_SRCS}
  ${ABSL_FLAGS_SRCS}
  spec_parser.h
  common.h
  util.h
  util.cc
  freelist.h
  filesystem.h
  filesystem.cc
  init.h
  error.cc
  trainer_factory.h
  trainer_factory.cc
  trainer_interface.h
  trainer_interface.cc
  bpe_model_trainer.h
  bpe_model_trainer.cc
  discretepiece_trainer.h
  discretepiece_trainer.cc
)

find_package(Threads REQUIRED)

list(APPEND SPM_LIBS ${PROTOBUF_LITE_LIBRARY} Threads::Threads)

if ((${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "mips") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "m68k") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "powerpc") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "sh4"))
  find_library(ATOMIC_LIB NAMES atomic libatomic.so libatomic.so.1)
  if (ATOMIC_LIB)
    message(STATUS "Found atomic: ${ATOMIC_LIB}")
    list(APPEND SPM_LIBS "atomic")
  endif()
endif()

add_library(sentencepiece_train-static STATIC ${SPM_TRAIN_SRCS})
set_target_properties(sentencepiece_train-static PROPERTIES OUTPUT_NAME "sentencepiece_train")

if (NOT MSVC)
  if (SPM_COVERAGE)
    set(CMAKE_CXX_FLAGS "-O0 -Wall -fPIC -coverage ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  endif()
  if (SPM_ENABLE_TENSORFLOW_SHARED)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  endif()
  if (SPM_NO_THREADLOCAL)
    add_definitions(-DSPM_NO_THREADLOCAL=1)
    add_definitions(-DGOOGLE_PROTOBUF_NO_THREADLOCAL=1)
  endif()
  set_source_files_properties(
    discretepiece.pb.cc discretepiece_model.pb.cc
    PROPERTIES COMPILE_FLAGS "-Wno-misleading-indentation")
  set_source_files_properties(${SPM_TEST_SRCS}
    PROPERTIES COMPILE_FLAGS "-Wno-sign-compare")
  if (SPM_ENABLE_SHARED)
    set_property(TARGET sentencepiece APPEND_STRING PROPERTY COMPILE_FLAGS " -DPIC")
    set_property(TARGET sentencepiece_train APPEND_STRING PROPERTY COMPILE_FLAGS " -DPIC")
  endif()
endif()

add_executable(spm_train spm_train_main.cc)
target_link_libraries(spm_train sentencepiece_train-static)



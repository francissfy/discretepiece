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

# flags
set(ABSL_FLAGS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/absl/flags/flag.cc)

# protobuf related
set(SPM_MODEL_PROTO_HDRS builtin_pb/discretepiece_model.pb.h)
set(SPM_MODEL_PROTO_SRCS builtin_pb/discretepiece_model.pb.cc)
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
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite/zero_copy_stream_impl_lite.cc
)

# shared files between train,encode,pybinding
set(SPM_SHARED_SRCS
  ${PROTOBUF_LITE_SRCS}
  ${ABSL_FLAGS_SRCS}
  builtin_pb/discretepiece_model.pb.h
  builtin_pb/discretepiece_model.pb.cc
  common.h
  util.h
  util.cc
  filesystem.h
  filesystem.cc
  init.h
  error.cc
)

# train sources
set(SPM_TRAIN_SRCS
  ${SPM_SHARED_SRCS}
  spec_parser.h
  freelist.h
  trainer_factory.h
  trainer_factory.cc
  trainer_interface.h
  trainer_interface.cc
  bpe_model_trainer.h
  bpe_model_trainer.cc
  discretepiece_trainer.h
  discretepiece_trainer.cc
)

# encode sources
set(SPM_ENCODE_SRCS
  ${SPM_SHARED_SRCS}
  model_interface.h
  model_interface.cc
  discretepiece_processor.h
  discretepiece_processor.cc
  model_factory.h
  model_factory.cc
  bpe_model.h
  bpe_model.cc
)

# converter sources
set(SPM_COMPATIBLE_CONVERTER_SRCS
  ${SPM_SHARED_SRCS}
  builtin_pb/sentencepiece_model.pb.h
  builtin_pb/sentencepiece_model.pb.cc
)

add_definitions(
  "-pthread 
  -DHAVE_PTHREAD=1 
  -Wno-sign-compare 
  -Wno-deprecated-declarations"
  "-D_USE_INTERNAL_STRING_VIEW"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/builtin_pb)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(Threads REQUIRED)
list(APPEND SPM_LIBS Threads::Threads)

# atomic libs
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

# train/encode static libs
add_library(train_static_lib STATIC ${SPM_TRAIN_SRCS})
set_target_properties(train_static_lib PROPERTIES OUTPUT_NAME "discretepiece_train")

add_library(encode_static_lib STATIC ${SPM_ENCODE_SRCS})
set_target_properties(encode_static_lib PROPERTIES OUTPUT_NAME "discretepiece_encode")

# install static lib too
list(APPEND SPM_INSTALLTARGETS train_static_lib encode_static_lib)

# train/encode shared libs, for python wrapper
if (SPM_ENABLE_SHARED)
  add_library(train_shared_lib SHARED ${SPM_TRAIN_SRCS})
  add_library(encode_shared_lib SHARED ${SPM_ENCODE_SRCS})
  list(APPEND SPM_INSTALLTARGETS train_shared_lib encode_shared_lib)
endif()


if (NOT MSVC)
  if (SPM_COVERAGE)
    set(CMAKE_CXX_FLAGS "-O0 -Wall -fPIC -coverage ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  endif()
  if (SPM_NO_THREADLOCAL)
    add_definitions(-DSPM_NO_THREADLOCAL=1)
    add_definitions(-DGOOGLE_PROTOBUF_NO_THREADLOCAL=1)
  endif()
  set_source_files_properties(
    discretepiece.pb.cc discretepiece_model.pb.cc
    PROPERTIES COMPILE_FLAGS "-Wno-misleading-indentation")
endif()

add_executable(spm_train spm_train_main.cc)
target_link_libraries(spm_train train_static_lib)

add_executable(spm_encode spm_encode_main.cc)
target_link_libraries(spm_encode encode_static_lib)

add_executable(spm_compatible_converter 
  ${SPM_COMPATIBLE_CONVERTER_SRCS}
  spm_compatible_converter.cc
)

# for install purpose
list(APPEND SPM_INSTALLTARGETS spm_encode spm_train spm_compatible_converter)

install(TARGETS ${SPM_INSTALLTARGETS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# install(FILES discretepiece_trainer.h discretepiece_processor.h DESTINATION ${CMAKE_INSTALL_INCDIR})
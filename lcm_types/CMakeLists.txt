find_package(lcm REQUIRED)
include(${LCM_USE_FILE})

lcm_wrap_types(
  C_EXPORT my_lcmtypes
  C_SOURCES c_sources
  C_HEADERS c_headers
  CPP_HEADERS cpp_headers
  contact_ground_truth_t.lcm
  contact_t.lcm
  leg_control_data_lcmt.lcm
  microstrain_lcmt.lcm
)

lcm_add_library(my_lcmtypes-cpp CPP ${cpp_headers})
target_include_directories(my_lcmtypes-cpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
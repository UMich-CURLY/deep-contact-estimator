# This Makefile was tested with GNU Make
CXX=g++

# Use pkg-config to lookup the proper compiler and linker flags for LCM
CFLAGS=`pkg-config --cflags lcm`
LDFLAGS=`pkg-config --libs lcm`

msg_types=../lcm_types/*.lcm

all: lcm-cnn-interface 

lcm-cnn-interface: lcm_cnn_interface.o 
	$(CXX) -o $@ $^ $(LDFLAGS)

# %.o: %.cpp ../lcm_types/cpp/contact_ground_truth_t.hpp
# 	$(CXX) $(CFLAGS) -I. -o $@ -c $<   

# ../lcm_types/cpp/%.hpp:
# 	lcm-gen -x ../lcm_types/*.lcm


clean:
	rm -f lcm-cnn-interface
	rm -f *.o

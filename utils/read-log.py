import sys
import lcm
sys.path.append("../lcm_types")
from python import contact_t
from python import leg_control_data_lcmt
from python import microstrain_lcmt

if len(sys.argv) < 2:
    sys.stderr.write("usage: read-log <logfile>\n")
    sys.exit(1)

log = lcm.EventLog(sys.argv[1], "r")
lc = lcm.LCM()

for event in log:
    if event.channel == "leg_control_data":
        leg_control_data_msg = leg_control_data_lcmt.decode(event.data)
        lc.publish("leg_control_data", leg_control_data_msg.encode())
    if event.channel == "microstrain":
        microstrain_msg = microstrain_lcmt.decode(event.data)
        lc.publish("microstrain", microstrain_msg.encode())
    # if event.channel == "contact_data":
    #     print("contact_data")
    #     contact_msg = contact_t.decode(event.data)
    #     lc.publish("contact_data", contact_msg.encode())
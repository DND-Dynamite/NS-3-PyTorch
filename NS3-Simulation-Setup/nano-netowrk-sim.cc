// #include "ns3/core-module.h"
// #include "ns3/network-module.h"
// #include "ns3/internet-module.h"
// #include "ns3/applications-module.h"
// #include "ns3/point-to-point-module.h"
// #include "ns3/flow-monitor-module.h"
// #include "ns3/thz-module.h" // THz channel models
// #include <python3.8/Python.h>
// #include <fstream>

// using namespace ns3;

// // Python integration for AI/ML
// class AIMLDetector {
// public:
//     AIMLDetector() {
//         Py_Initialize();
//         PyRun_SimpleString("import sys");
//         PyRun_SimpleString("sys.path.append('./')");
        
//         PyObject* pModule = PyImport_ImportModule("dos_detector");
//         if (pModule != NULL) {
//             pDetectFunc = PyObject_GetAttrString(pModule, "detect_dos_attack");
//         }
//     }
    
//     ~AIMLDetector() {
//         Py_Finalize();
//     }
    
//     bool analyzeTraffic(double packetRate, double dataSize, double errorRate) {
//         if (pDetectFunc && PyCallable_Check(pDetectFunc)) {
//             PyObject* pArgs = PyTuple_Pack(3, 
//                 PyFloat_FromDouble(packetRate),
//                 PyFloat_FromDouble(dataSize),
//                 PyFloat_FromDouble(errorRate));
            
//             PyObject* pResult = PyObject_CallObject(pDetectFunc, pArgs);
//             bool isAttack = PyObject_IsTrue(pResult);
            
//             Py_DECREF(pArgs);
//             Py_DECREF(pResult);
//             return isAttack;
//         }
//         return false;
//     }

// private:
//     PyObject* pDetectFunc;
// };

// // Traffic monitor for collecting simulation data
// class TrafficMonitor {
// public:
//     void RecordPacket(Ptr<const Packet> packet, const Address& from) {
//         totalPackets++;
//         currentSecondPackets++;
        
//         // Record for ML features
//         packetSizes.push_back(packet->GetSize());
//         packetTimestamps.push_back(Simulator::Now().GetSeconds());
        
//         // Analyze every 100 packets
//         if (totalPackets % 100 == 0) {
//             ExtractFeatures();
//         }
//     }
    
//     void ExtractFeatures() {
//         if (packetTimestamps.size() < 2) return;
        
//         double timeWindow = packetTimestamps.back() - packetTimestamps.front();
//         double packetRate = packetSizes.size() / timeWindow;
        
//         double totalSize = 0;
//         for (size_t i = 0; i < packetSizes.size(); i++) {
//             totalSize += packetSizes[i];
//         }
//         double avgPacketSize = totalSize / packetSizes.size();
        
//         // Send to AI detector
//         bool isAttack = aiDetector.analyzeTraffic(packetRate, avgPacketSize, 0.0);
        
//         if (isAttack) {
//             std::cout << "🚨 AI DETECTED DoS ATTACK at " << Simulator::Now().GetSeconds() 
//                       << "s - Rate: " << packetRate << " pkts/s" << std::endl;
//             TriggerMitigation();
//         }
        
//         // Clear for next window
//         packetSizes.clear();
//         packetTimestamps.clear();
//     }
    
//     void TriggerMitigation() {
//         std::cout << "[MITIGATION] Activating rate limiting and source blocking" << std::endl;
//         // Implement NS-3 mitigation strategies here
//     }

// private:
//     uint32_t totalPackets = 0;
//     uint32_t currentSecondPackets = 0;
//     std::vector<uint32_t> packetSizes;
//     std::vector<double> packetTimestamps;
//     AIMLDetector aiDetector;
// };

// int main(int argc, char *argv[]) {
//     // NS-3 simulation parameters
//     uint32_t nanoNodes = 50;
//     uint32_t maliciousNodes = 5;
//     double simulationTime = 60.0;
    
//     // Create nano nodes
//     NodeContainer nanoNetwork;
//     nanoNetwork.Create(nanoNodes);
    
//     // Configure THz channel for nano communication
//     THzHelper thz;
//     thz.SetChannelModel("MolecularAbsorption");
//     thz.SetMacType("NanoMac");
    
//     NetDeviceContainer nanoDevices = thz.Install(nanoNetwork);
    
//     // Internet stack
//     InternetStackHelper internet;
//     internet.Install(nanoNetwork);
    
//     // Assign IP addresses
//     Ipv4AddressHelper ipv4;
//     ipv4.SetBase("10.1.1.0", "255.255.255.0");
//     Ipv4InterfaceContainer interfaces = ipv4.Assign(nanoDevices);
    
//     // Install applications - normal traffic
//     uint16_t port = 9;
//     OnOffHelper onOffHelper("ns3::UdpSocketFactory", 
//                            InetSocketAddress(interfaces.GetAddress(1), port));
//     onOffHelper.SetConstantRate(DataRate("100kbps"));
//     onOffHelper.SetAttribute("PacketSize", UintegerValue(512));
    
//     ApplicationContainer normalApps = onOffHelper.Install(nanoNetwork.Get(0));
//     normalApps.Start(Seconds(1.0));
//     normalApps.Stop(Seconds(simulationTime - 1));
    
//     // Malicious nodes - DoS attackers
//     OnOffHelper dosHelper("ns3::UdpSocketFactory", 
//                          InetSocketAddress(interfaces.GetAddress(1), port));
//     dosHelper.SetConstantRate(DataRate("10Mbps")); // High rate for DoS
//     dosHelper.SetAttribute("PacketSize", UintegerValue(1024));
    
//     for (uint32_t i = nanoNodes - maliciousNodes; i < nanoNodes; ++i) {
//         ApplicationContainer dosApps = dosHelper.Install(nanoNetwork.Get(i));
//         dosApps.Start(Seconds(20.0)); // Start attack at 20 seconds
//         dosApps.Stop(Seconds(40.0));  // Stop attack at 40 seconds
//     }
    
//     // Install traffic monitor
//     Ptr<TrafficMonitor> monitor = Create<TrafficMonitor>();
    
//     // Connect trace sources for packet monitoring
//     for (uint32_t i = 0; i < nanoNodes; ++i) {
//         Ptr<NetDevice> device = nanoDevices.Get(i);
//         device->TraceConnectWithoutContext("MacTx", 
//             MakeCallback(&TrafficMonitor::RecordPacket, monitor));
//     }
    
//     // Flow monitor for analysis
//     FlowMonitorHelper flowmon;
//     Ptr<FlowMonitor> monitorPtr = flowmon.InstallAll();
    
//     std::cout << "Starting NS-3 Nano Network Simulation with AI/ML DoS Detection..." << std::endl;
//     Simulator::Stop(Seconds(simulationTime));
//     Simulator::Run();
    
//     // Analysis output
//     monitorPtr->CheckForLostPackets();
//     Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
//     FlowMonitor::FlowStatsContainer stats = monitorPtr->GetFlowStats();
    
//     for (auto it = stats.begin(); it != stats.end(); ++it) {
//         Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
//         std::cout << "Flow " << it->first << " (" << t.sourceAddress << " -> " 
//                   << t.destinationAddress << ")\n";
//         std::cout << "  Tx Packets: " << it->second.txPackets << "\n";
//         std::cout << "  Rx Packets: " << it->second.rxPackets << "\n";
//         std::cout << "  Packet Loss: " << it->second.lostPackets << "\n";
//         std::cout << "  Throughput: " << it->second.rxBytes * 8.0 / simulationTime / 1000 
//                   << " kbps\n";
//     }
    
//     Simulator::Destroy();
//     return 0;
// }
#include <cstdio>
#include <iostream>
#include <sys/socket.h>
#include <sstream>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <memory>

#include "ace/OS.h"
#include "ace/ACE.h"
#include "ace/Process.h"
#include "ace/Process_Mutex.h"
#include "ace/Guard_T.h"

#include "GEtypes.h"
#include "cnv_endian_API.h"

#include "mqueue.h"
#include "rc_os.h"

#include "RtpCommon.h"
#include "RtpDemoIF.h"
#include "RtpAps.h"

// Example for MDArray, Fourier and Hdf5
#include <MDArray/MDArray.h>
#include <MDArray/Fourier.h>
#include <Hdf5/File.h>

static mqd_t app_qid;
static mqd_t server_qid;
static RtpFrameDataStruct* pshmRtpFrameDataStruct;
static int server_socket, client_socket;

/*
 * Buffer to hold a single frame (all channels).
 */
constexpr auto MAX_FRAME_SIZE = 512U;
static MDArray::ComplexFloatMatrix currDataArray;

static auto debugFlag = true;
static auto feedbackSizeInBytes = 0U;
static auto dataS32 = 0;
static auto dataF32 = 0.0f;
static std::vector<float> floatValVector;

std::unique_ptr<GEHdf5::File> debugHdf;

/* --- Internal Function Prototypes --- */
static void rtpInit();
static void rtpProcessData(const RtpDataMessageStruct &dataMsg);
static void rtpEnd();
static void fillProcessedResult(RtpResultStruct &result, const float floatVal);
static void traceMessage(const std::string &message, const bool logFlag = true);
static void cnv_endian_RtpDemoResult(RtpDemoResult *_ecnv_target_);
static void setupTcpServer();
static void sendCurrDataArrayToClient(const MDArray::ComplexFloatMatrix &data);

/**
 * callback for RTP_INIT opcode.  Initializes RTP application.
 */
static void rtpInit()
{
    floatValVector.clear();
    return;
}

/**
 * callback for RTP_DATA opcode.  Copies data and sends counter to PSD.
 *
 * @param[in] dataMsg - Data message for current frame
 */
static void rtpProcessData(const RtpDataMessageStruct &dataMsg)
{
    // add lock 
    ACE_Process_Mutex serverAppMutex("rtpServerAppSynch");

    traceMessage("timestampUs=" + std::to_string(dataMsg.acqTimestampUs));

    float floatVal = 0.0f;
    const auto expectedSizeInBytes = currDataArray.numElements() * sizeof(MDArray::ComplexFloat);

    /* Copy data to a local buffer */
    if ((size_t)(dataMsg.sizeInBytesOfFrame) == expectedSizeInBytes)
    {
        /*
         * Begin ACE_Guard scope.  ACE_Guard synchronizes access of
         * RtpFrameDataStruct in shared memory between RTP Server and
         * RTP App
         */
        ACE_Guard<ACE_Process_Mutex> guard(serverAppMutex);
        // copy 2 to 1 ,length is 3
        memcpy(currDataArray.data(), (pshmRtpFrameDataStruct->dataPointsBuf)[dataMsg.ringBufIndex], dataMsg.sizeInBytesOfFrame);
        MDArray::Fourier::Fft(currDataArray, MDArray::firstDim);
        std::cout << "currDataArray = " << currDataArray << std::endl;
        std::cout << "currDataArray.data() = " << currDataArray.data() << std::endl;
        floatVal = sum(abs(currDataArray));
        // for demo purpose, use std::cout to print. In general, traceMessage is the preferred way.
        std::cout << "floatVal = " << floatVal << std::endl;

        // save it temporarily for debug, see line 129
        floatValVector.emplace_back(floatVal);

        // Send currDataArray to client
        sendCurrDataArrayToClient(currDataArray);
    }
    else
    {
        /* Frame too large for buffer.  Skip copy and log */
        std::stringstream msg;
        msg << "Received data size (" << dataMsg.sizeInBytesOfFrame << ") does not match expected size(" << expectedSizeInBytes << ").  Skipping frame copy";
        traceMessage(msg.str());
    }

    /* Populate and send counter as the result */
    RtpProcessedResultsPkt procResPkt;
    procResPkt.hdr.opcode = SEND_FEEDBACK_AUTO;
    fillProcessedResult(procResPkt.rtpResults, floatVal);
    msgSend(app_qid, reinterpret_cast<char *>(&(procResPkt)), sizeof(procResPkt), 0);

    return;
}

/**
 * callback for RTP_END opcode.  Cleanup when exiting application.
 */
static void rtpEnd()
{
    if (debugHdf)
    {
        debugHdf->Write(floatValVector, "debugData");
        debugHdf->Close();
    }
    return;
}

/**
 * fills the RtpResultStruct structure with the processed results
 *
 * @param[out] result - Reference to RtpResultStruct
 */
static void fillProcessedResult(RtpResultStruct &result, const float floatVal)
{
    RtpDemoResult demoResult;

    result.packedResult = 0; // not used
    result.packedResultValid = 0;
    memset(&demoResult, 0, sizeof(demoResult));

    demoResult.s32data = dataS32;
    demoResult.f32data = dataF32;
    demoResult.floatVal = floatVal;

    // Use unpacked pathway
    result.unpackedResult.opcode = RTP_RESULT_DEMO_UNPACKED;
    result.unpackedResultSize = feedbackSizeInBytes + sizeof(result.unpackedResult.opcode);
    cnv_endian_RtpDemoResult(&demoResult);

    // copy feedbackSizeInBytes length data to unpackedResult
    memcpy((void *)result.unpackedResult.data, (void *)&demoResult, feedbackSizeInBytes);

    return;
}

void cnv_endian_RtpDemoResult(RtpDemoResult *_ecnv_target_)
{
    cnv_endian_s32(&_ecnv_target_->s32data);
    cnv_endian_f32(&_ecnv_target_->f32data);
    cnv_endian_f32(&_ecnv_target_->floatVal);

    for (auto ii = 0U; ii < RTP_DEMO_RESULT_DATA_ARRAY_SIZE; ++ii)
    {
        cnv_endian_s32(&_ecnv_target_->data[ii]);
    }
}

/**
 * Logs message to RTP server if logFlag is true
 *
 * @param[in] message - Message to log
 * @param[in] logFlag - Flag to determine whether to log the message
 */
static void traceMessage(const std::string &message, const bool logFlag)
{
    if (logFlag)
    {
        static RtpAppTraceLogPkt tracePkt;
        tracePkt.hdr.opcode = TRACE_OR_LOG_RTP_APP_MESSAGES;
        tracePkt.rtpTraceLog.subOpcode = USE_RTP_TRACE;

        strncpy(tracePkt.rtpTraceLog.trace, message.c_str(), sizeof(tracePkt.rtpTraceLog.trace));
        tracePkt.rtpTraceLog.trace[sizeof(tracePkt.rtpTraceLog.trace) - 1] = 0;

        msgSend(app_qid, reinterpret_cast<char *>(&tracePkt), sizeof(tracePkt), 0);
    }
}

/**
 * Set up a TCP server to send currDataArray to the client.
 */
static void setupTcpServer()
{
    struct sockaddr_in server_addr;

    // Create server socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0)
    {
        perror("Failed to create socket");
        traceMessage("ltbg=========================Failed to create socket", debugFlag);
	    std::cout << "failed to create socket." << std::endl;
       	exit(1);
    }
    std::cout << "ltbg=========================create socket sucessfully." << std::endl;

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);  // Listen on port 12345
    server_addr.sin_addr.s_addr = inet_addr("10.30.1.41");

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("Bind failed");
        traceMessage("ltbg===================Bind failed", debugFlag);
        std::cout << "bind failed with errno=" << errno << std::endl;
        close(server_socket);
        exit(1);
    }
    std::cout << "ltbg=========================bind sucessfully." << std::endl;



    if (listen(server_socket, 3) < 0)
    {
        perror("Listen failed");
        traceMessage("ltbg=================Listen failed", debugFlag);

        std::cout << "listen failed= " << client_socket << std::endl;
        std::cout << "listen failed with errno=" << errno << std::endl;
        close(server_socket);
        exit(1);
    }
    std::cout << "Server listening on port 8080..." << std::endl;
    
    client_socket = accept(server_socket, NULL, NULL);
    if (client_socket < 0)
    {
        perror("Accept failed");
        traceMessage("ltbg=================Accept failed", debugFlag);
        std::cout << "accept failed" << std::endl;
        close(server_socket);
        exit(1);
    }

    std::cout << "client_socket= " << client_socket << std::endl;
    traceMessage("ltbg=================client_socket=", debugFlag);


}

/**
 * Send currDataArray to the client via TCP.
 */
static void sendCurrDataArrayToClient(const MDArray::ComplexFloatMatrix &data)
{
    ssize_t bytes_sent = send(client_socket, data.data(), data.numElements() * sizeof(MDArray::ComplexFloat), 0);
    if (bytes_sent < 0)
    {
        perror("Send failed");
        traceMessage("ltbg=================Send failed", debugFlag);
	    std::cout << "sen failed" << std::endl;
        close(client_socket);
        exit(1);
    }
    std::cout << "Sent currDataArray to client." << std::endl;
    traceMessage("ltbg=================Sent currDataArray to client.");

}

int main()
{
    RtpDataValuesPkt *pshmRtpDataVal;
    RtpDataMessageStruct *pRtpDataMsg;
    RtpDataMessageStruct rtpDataMsg;

    /*
     * Attach to shared memory created by RTP Server and setup pointer
     * to RtpDataValuesPkt and RtpFrameDataStruct structures in Shared
     * Memory.
     */
    void *baseAddr = openSharedMemory(SHARED_MEMORY_LOCATION, SHARED_MEMORY_SIZE, 0);
    if (baseAddr == NULL)
    {
        perror("Failed to open shared memory");
        std::cout << "ltbg=================Failed to open shared memory." << std::endl;
        traceMessage("ltbg=================Failed to open shared memory", baseAddr);

        exit(1);
    }
    pshmRtpDataVal = static_cast<RtpDataValuesPkt *>(baseAddr);
    void *nextAddr = (void *)((uintptr_t)baseAddr + sizeof(RtpDataValuesPkt));
    pshmRtpFrameDataStruct = static_cast<RtpFrameDataStruct *>(nextAddr);

    /* Copy values from RtpDataValuesPkt struct */
    debugFlag = pshmRtpDataVal->rtpDataVal.vreDebug;

    //if (debugFlag)
    //{
    //std::ofstream debugTxt("/usr/g/bin/RtpDebug.txt", std::ios::out);
    //if (!debugTxt)
    //{
    //    throw std::ios_base::failure("Failed to open the text file for writing");
   // }
   // debugTxt << "Debug Data\n";  // Optional: 添加文件头部信息
   // debugTxt.close();
   // }

    currDataArray.resize(pshmRtpDataVal->rtpDataVal.frameSize, pshmRtpDataVal->rtpDataVal.numRtpReceivers);
    currDataArray = 0;

    feedbackSizeInBytes = pshmRtpDataVal->rtpDataVal.intVar_1;
    if (feedbackSizeInBytes > RTP_UNPACKED_RESULT_SIZE)
    {
        perror("feedback size in bytes too large");
        std::cout << "ltbg=================feedback size in bytes too large." << std::endl;
        traceMessage("ltbg=================feedback size in bytes too large", debugFlag);

        exit(1);
    }

    dataS32 = pshmRtpDataVal->rtpDataVal.intVar_2;
    dataF32 = pshmRtpDataVal->rtpDataVal.floatVar_2;

    traceMessage("Attached to shared memory", debugFlag);
    traceMessage("ltbg=================Attached to shared memory", debugFlag);
    std::cout << "ltbg=================Attached to shared memory." << std::endl;


    /* Open message queues created by RTP Server */
    app_qid = mq_open(APPtoSERVER_MSGQOBJ_NAME, O_RDWR);
    if (app_qid == (mqd_t)-1)
    {
        perror("In mq_open()");
        traceMessage("In mq_open()", debugFlag);
        std::cout << "ltbg================= error app_qid In mq_open()." << std::endl;
        exit(1);
    }

    server_qid = mq_open(SERVERtoAPP_MSGQOBJ_NAME, O_RDWR);
    if (server_qid == (mqd_t)-1)
    {
        perror("In mq_open()");
        traceMessage("In mq_open()", debugFlag);
        std::cout << "ltbg================= error server_qid In mq_open()." << std::endl;
        exit(1);
    }

    traceMessage("Opened message queues", debugFlag);

    traceMessage("ltbg=========Opened message queues", debugFlag);
    std::cout << "ltbg================= Opened message queues" << std::endl;

    // Setup TCP server to send currDataArray to client
    setupTcpServer();
    std::cout << "ltbg================= end setupTCP" << std::endl;


    /* Wait for messages until RTP_END */
    char msgRecBuf[256];
    int *receivedOpcode;
    traceMessage("*******************ltbg=========== testrun to int *receivedOpcode", debugFlag);
    std::cout << "*******************ltbg=========== testrun to int *receivedOpcode" << std::endl;

    auto taskEnd = false;
    while (!taskEnd)
    {
        msgRecv(server_qid, msgRecBuf, 100, -1);
        receivedOpcode = reinterpret_cast<int *>(msgRecBuf);
    traceMessage("*******************ltbg test receivedOpcode", debugFlag);
    std::cout << "receivedOpcode= " << receivedOpcode << std::endl;

        switch (*receivedOpcode)
        {
        case RTP_INIT:
            traceMessage("RTP_INIT received", debugFlag);
            traceMessage("*******************ltbg=========== RTP_INIT received", debugFlag);
            std::cout << "*******************ltbg=========== RTP_INIT received" << std::endl;

            rtpInit();
            break;

        case RTP_DATA:
            traceMessage("RTP_DATA received", debugFlag);
            traceMessage("*******************ltbg=========== RTP_DATA received", debugFlag);
            std::cout << "*******************ltbg=========== RTP_DATA received" << std::endl;

            pRtpDataMsg = reinterpret_cast<RtpDataMessageStruct *>(msgRecBuf);
            rtpDataMsg = *pRtpDataMsg;
            rtpProcessData(rtpDataMsg);
            break;

        case RTP_END:
            traceMessage("RTP_END received", debugFlag);
            traceMessage("*******************ltbg=========== RTP_END received", debugFlag);
            std::cout << "*******************ltbg=========== RTP_END received" << std::endl;

            rtpEnd();
            taskEnd = true;
            break;

        default:
            std::stringstream msg;
            msg << "Unknown opcode " << *receivedOpcode << " received";
            traceMessage(msg.str());
            std::cout << "*******************ltbg=========== Unknown opcode received" << std::endl;

            break;
        }
    }

    close(client_socket);
    close(server_socket);
    std::cout << "*******************ltbg=========== close socket" << std::endl;



    return 0;
}

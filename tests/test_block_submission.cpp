#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mocks.hpp"
#include "mpi_helpers.hpp"
#include "restore/block_serialization.hpp"
#include "restore/block_submission.hpp"
#include "restore/common.hpp"
#include "restore/core.hpp"

using namespace ::testing;

TEST(BlockSubmissionTest, exchangeData) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStoreMPI::RecvMessage;
    using ReStoreMPI::SendMessage;
    using SendBuffers = ReStore::BlockSubmissionCommunication<std::byte>::SendBuffers;

    std::vector<RecvMessage> dummyReceive;

    {
        SendBuffers       emptySendBuffers(10);
        MPIContextMock    mpiContext;
        BlockDistribution blockDistribution(10, 100, 3, mpiContext);

        ReStore::BlockSubmissionCommunication<uint8_t, MPIContextMock> comm(
            mpiContext, blockDistribution,
            ReStore::OffsetModeDescriptor{ReStore::OffsetMode::constant, sizeof(uint8_t)});

        std::vector<SendMessage> expectedSendMessages{{}};
#ifndef DENSE_ALL_TO_ALL_IN_SUBMIT_BLOCKS
        EXPECT_CALL(mpiContext, SparseAllToAll(Eq(expectedSendMessages))).WillOnce(Return(dummyReceive));
#else
        EXPECT_CALL(mpiContext, getCurrentSize()).WillOnce(Return(10));
        EXPECT_CALL(
            mpiContext, alltoall(
                            UnorderedElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                            UnorderedElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1));
#endif
        ASSERT_NO_THROW(comm.exchangeData(emptySendBuffers));
    }

    {
        MPIContextMock    mpiContext;
        BlockDistribution blockDistribution(10, 100, 3, mpiContext);

        ReStore::BlockSubmissionCommunication<uint8_t, MPIContextMock> comm(
            mpiContext, blockDistribution,
            ReStore::OffsetModeDescriptor{ReStore::OffsetMode::constant, sizeof(uint8_t)});

        SendBuffers sendBuffers(10);
        sendBuffers[0] = {0_byte, 1_byte, 2_byte, 3_byte};
        sendBuffers[1] = {4_byte, 5_byte, 6_byte, 7_byte};

        std::vector<SendMessage> expectedSendMessages{{sendBuffers[0].data(), 4, 0}, {sendBuffers[1].data(), 4, 1}};

#ifndef DENSE_ALL_TO_ALL_IN_SUBMIT_BLOCKS
        EXPECT_CALL(mpiContext, SparseAllToAll(UnorderedElementsAreArray(expectedSendMessages)))
            .WillOnce(Return(dummyReceive));
#else
        EXPECT_CALL(mpiContext, getCurrentSize()).WillOnce(Return(10));
        EXPECT_CALL(
            mpiContext, alltoall(
                            UnorderedElementsAre(4, 4, 0, 0, 0, 0, 0, 0, 0, 0),
                            UnorderedElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1));
#endif
        EXPECT_EQ(comm.exchangeData(sendBuffers), dummyReceive);
    }
}

TEST(BlockSubmissionTest, ParseIncomingMessages) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::RecvMessage;

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(3));

    BlockDistribution blockDistribution(10, 100, 3, mpiContext);

    ReStore::BlockSubmissionCommunication<uint16_t, MPIContextMock> comm(
        mpiContext, blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, sizeof(uint16_t)});

    RecvMessage message1(
        std::vector<std::byte>{
            // We have to write everything in big endian notation
            1_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // id 1 ...
            1_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to 1
            0x02_byte, 0x02_byte,                                                 // id: 1, payload 0x0202
            3_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // id 3 ...
            3_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to 3
            0x12_byte, 0x23_byte                                                  // id: 3, payload 0x3412
        },
        0);

    RecvMessage message2(
        std::vector<std::byte>{
            0_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // id 0 ...
            0_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to 0
            0x37_byte, 0x13_byte,                                                 // payload 0x1337
            8_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // id 8 ...
            8_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to 8
            0x42_byte, 0x00_byte,                                                 // payload 0x0042
            6_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // id 6 ...
            6_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to 6
            0x11_byte, 0x11_byte                                                  // payload 0x1111
        },
        1);

    RecvMessage message3(
        std::vector<std::byte>{
            0_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // id 0 ...
            4_byte,    0_byte,    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to 4
            0x02_byte, 0x00_byte,                                                 // payload 2
            0x03_byte, 0x00_byte,                                                 // payload 3
            0x04_byte, 0x00_byte,                                                 // payload 4
            0x05_byte, 0x00_byte,                                                 // payload 5
            0x06_byte, 0x00_byte                                                  // payload 6
        },
        2);


    auto called = 0;
    comm.parseIncomingMessage(
        message1,
        [&called](ReStore::block_id_t blockId, const std::byte* data, size_t lengthInBytes, current_rank_t srcRank) {
            if (called == 0) {
                ASSERT_EQ(blockId, 1);
                ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x0202);
                ASSERT_EQ(lengthInBytes, 2);
                ASSERT_EQ(srcRank, 0);
            } else if (called == 1) {
                ASSERT_EQ(blockId, 3);
                ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x2312);
                ASSERT_EQ(lengthInBytes, 2);
                ASSERT_EQ(srcRank, 0);
            } else {
                FAIL();
            }
            ++called;
        });
    ASSERT_EQ(called, 2);

    called = 0;
    comm.parseIncomingMessage(
        message2,
        [&called](ReStore::block_id_t blockId, const std::byte* data, size_t lengthInBytes, current_rank_t srcRank) {
            switch (called) {
                case 0:
                    ASSERT_EQ(blockId, 0);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x1337);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 1);
                    break;
                case 1:
                    ASSERT_EQ(blockId, 8);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x0042);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 1);
                    break;
                case 2:
                    ASSERT_EQ(blockId, 6);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x1111);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 1);
                    break;
                default:
                    FAIL();
            }
            ++called;
        });
    ASSERT_EQ(called, 3);

    called = 0;
    comm.parseIncomingMessage(
        message3,
        [&called](ReStore::block_id_t blockId, const std::byte* data, size_t lengthInBytes, current_rank_t srcRank) {
            ASSERT_EQ(blockId, called);
            ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), called + 2);
            ASSERT_EQ(lengthInBytes, 2);
            ASSERT_EQ(srcRank, 2);
            ++called;
        });
    ASSERT_EQ(called, 5);

    called = 0;
    std::vector<RecvMessage> messages{message1, message2, message3};
    comm.parseAllIncomingMessages(
        messages,
        [&called](ReStore::block_id_t blockId, const std::byte* data, size_t lengthInBytes, current_rank_t srcRank) {
            switch (called) {
                case 0:
                    ASSERT_EQ(blockId, 1);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x0202);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 0);
                    break;
                case 1:
                    ASSERT_EQ(blockId, 3);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x2312);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 0);
                    break;
                case 2:
                    ASSERT_EQ(blockId, 0);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x1337);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 1);
                    break;
                case 3:
                    ASSERT_EQ(blockId, 8);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x0042);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 1);
                    break;
                case 4:
                    ASSERT_EQ(blockId, 6);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 0x1111);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 1);
                    break;
                case 5:
                    ASSERT_EQ(blockId, 0);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 2);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 2);
                    break;
                case 6:
                    ASSERT_EQ(blockId, 1);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 3);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 2);
                    break;
                case 7:
                    ASSERT_EQ(blockId, 2);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 4);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 2);
                    break;
                case 8:
                    ASSERT_EQ(blockId, 3);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 5);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 2);
                    break;
                case 9:
                    ASSERT_EQ(blockId, 4);
                    ASSERT_EQ(*reinterpret_cast<const uint16_t*>(data), 6);
                    ASSERT_EQ(lengthInBytes, 2);
                    ASSERT_EQ(srcRank, 2);
                    break;
                default:
                    FAIL();
            }
            ++called;
        });
    ASSERT_EQ(called, 10);
}

TEST(BlockSubmissionTest, CopyAlreadySerializedBlocksToSendBuffers) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;

    struct World {
        bool    useMagic;
        uint8_t unicornCount;
    };

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(10));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(10));

    auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);

    ReStore::BlockSubmissionCommunication<World, MPIContextMock> comm(
        mpiContext, *blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, 2});

    World              earth       = {false, 0};
    World              narnia      = {true, 10};
    World              middleEarth = {true, 0};
    std::vector<World> worlds      = {earth, narnia, middleEarth};

    std::vector<std::byte> serializedData = {
        0_byte,  0_byte, // earth
        10_byte, 1_byte, // narnia
        0_byte,  1_byte  // middle earth
    };
    const size_t localNumberOfBlocks = 3;

    std::vector<ReStore::SerializedBlocksDescriptor> blockDescriptors = {{0, 3, serializedData.data()}};
    auto sendBuffers = comm.copySerializedBlocksToSendBuffers(blockDescriptors, localNumberOfBlocks);

    // All these three blocks belong to range 0 and are therefore stored on ranks 0, 3 and 6
    std::vector<std::byte> expectedSendBuffer = {
        0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        2_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 2
    };
    expectedSendBuffer.insert(expectedSendBuffer.end(), serializedData.begin(), serializedData.end());

    ASSERT_EQ(sendBuffers.size(), 10);

    ASSERT_EQ(sendBuffers[0].size(), 22);
    ASSERT_EQ(sendBuffers[3].size(), 22);
    ASSERT_EQ(sendBuffers[6].size(), 22);

    ASSERT_EQ(sendBuffers[0], expectedSendBuffer);
    ASSERT_EQ(sendBuffers[3], expectedSendBuffer);
    ASSERT_EQ(sendBuffers[6], expectedSendBuffer);
}

TEST(BlockSubmissionTest, CopyAlreadySerializedBlocksToSendBuffers_rangeSpanningMultiplePEs) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;
    const size_t   numberOfPEs      = 10;
    const uint16_t replicationLevel = 3;
    const size_t   numberOfBlocks   = 20;

    struct World {
        bool    useMagic;
        uint8_t unicornCount;
    };

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(numberOfPEs));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(numberOfPEs));

    auto blockDistribution =
        std::make_shared<BlockDistribution>(numberOfPEs, numberOfBlocks, replicationLevel, mpiContext);

    ReStore::BlockSubmissionCommunication<World, MPIContextMock> comm(
        mpiContext, *blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, 2});

    World              earth       = {false, 0};
    World              narnia      = {true, 10};
    World              middleEarth = {true, 0};
    std::vector<World> worlds      = {earth, narnia, middleEarth};

    std::vector<std::byte> serializedData = {
        0_byte,  0_byte, // earth
        10_byte, 1_byte, // narnia
        0_byte,  1_byte  // middle earth
    };
    const size_t localNumberOfBlocks = 3;

    std::vector<ReStore::SerializedBlocksDescriptor> blockDescriptors = {{0, 3, serializedData.data()}};
    auto sendBuffers = comm.copySerializedBlocksToSendBuffers(blockDescriptors, localNumberOfBlocks);

    // There are two blocks per range. Therefore, blocks 0 and 1 are stored on ranks 0, 3, and 6. Block 2 is stored on
    // ranks 1, 4, and 7.
    std::vector<std::byte> expectedSendBuffer036 = {
        0_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        1_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 1
        0_byte,  0_byte,                                                 // earth
        10_byte, 1_byte,                                                 // narnia
    };

    std::vector<std::byte> expectedSendBuffer147 = {
        2_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 2
        2_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 2
        0_byte, 1_byte,                                                 // middle earth
    };

    ASSERT_EQ(sendBuffers.size(), 10);

    ASSERT_EQ(sendBuffers[0].size(), 20);
    ASSERT_EQ(sendBuffers[3].size(), 20);
    ASSERT_EQ(sendBuffers[6].size(), 20);

    ASSERT_EQ(sendBuffers[0], expectedSendBuffer036);
    ASSERT_EQ(sendBuffers[3], expectedSendBuffer036);
    ASSERT_EQ(sendBuffers[6], expectedSendBuffer036);

    ASSERT_EQ(sendBuffers[1].size(), 18);
    ASSERT_EQ(sendBuffers[4].size(), 18);
    ASSERT_EQ(sendBuffers[7].size(), 18);

    ASSERT_EQ(sendBuffers[1], expectedSendBuffer147);
    ASSERT_EQ(sendBuffers[4], expectedSendBuffer147);
    ASSERT_EQ(sendBuffers[7], expectedSendBuffer147);
}

TEST(BlockSubmissionTest, CopyAlreadySerializedBlocksToSendBuffers_multipleRangesSpanningMultiplePEs) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;
    const size_t   numberOfPEs      = 10;
    const uint16_t replicationLevel = 3;
    const size_t   numberOfBlocks   = 20;

    struct World {
        bool    useMagic;
        uint8_t unicornCount;
    };

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(numberOfPEs));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(numberOfPEs));

    auto blockDistribution =
        std::make_shared<BlockDistribution>(numberOfPEs, numberOfBlocks, replicationLevel, mpiContext);

    ReStore::BlockSubmissionCommunication<World, MPIContextMock> comm(
        mpiContext, *blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, 2});

    std::vector<std::byte> serializedData = {
        0_byte, 0_byte,  // earth; block 0
        10_byte, 1_byte, // narnia; block 1
        // Block 2 not serialized!
        0_byte, 1_byte // middle earth; block3
    };
    const size_t localNumberOfBlocks = 3;

    std::vector<ReStore::SerializedBlocksDescriptor> blockDescriptors = {
        {0, 2, serializedData.data()},
        {3, 4, serializedData.data() + 2 * sizeof(World)}};
    auto sendBuffers = comm.copySerializedBlocksToSendBuffers(blockDescriptors, localNumberOfBlocks);

    // There are two blocks per range. Therefore, blocks 0 and 1 are stored on ranks 0, 3, and 6. Block 3 is stored on
    // ranks 1, 4, and 7.
    std::vector<std::byte> expectedSendBuffer036 = {
        0_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        1_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 1
        0_byte,  0_byte,                                                 // earth
        10_byte, 1_byte,                                                 // narnia
    };

    std::vector<std::byte> expectedSendBuffer147 = {
        3_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 3
        3_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 3
        0_byte, 1_byte,                                                 // middle earth
    };

    ASSERT_EQ(sendBuffers.size(), 10);

    ASSERT_EQ(sendBuffers[0].size(), 20);
    ASSERT_EQ(sendBuffers[3].size(), 20);
    ASSERT_EQ(sendBuffers[6].size(), 20);

    ASSERT_EQ(sendBuffers[0], expectedSendBuffer036);
    ASSERT_EQ(sendBuffers[3], expectedSendBuffer036);
    ASSERT_EQ(sendBuffers[6], expectedSendBuffer036);

    ASSERT_EQ(sendBuffers[1].size(), 18);
    ASSERT_EQ(sendBuffers[4].size(), 18);
    ASSERT_EQ(sendBuffers[7].size(), 18);

    ASSERT_EQ(sendBuffers[1], expectedSendBuffer147);
    ASSERT_EQ(sendBuffers[4], expectedSendBuffer147);
    ASSERT_EQ(sendBuffers[7], expectedSendBuffer147);
}

TEST(BlockSubmissionTest, CopyAlreadySerializedBlocksToSendBuffers_multipleRangesSpanningMultiplePEs2) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;
    const size_t   numberOfPEs      = 10;
    const uint16_t replicationLevel = 3;
    const size_t   numberOfBlocks   = 40;

    struct World {
        bool    useMagic;
        uint8_t unicornCount;
    };

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(numberOfPEs));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(numberOfPEs));

    auto blockDistribution =
        std::make_shared<BlockDistribution>(numberOfPEs, numberOfBlocks, replicationLevel, mpiContext);

    ReStore::BlockSubmissionCommunication<World, MPIContextMock> comm(
        mpiContext, *blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, 2});

    std::vector<std::byte> serializedData = {
        0_byte,  0_byte, // earth
        10_byte, 1_byte, // narnia
        0_byte,  1_byte, // middle earth; not transmitted
        1_byte,  1_byte, // narnia1; not transmitted
        2_byte,  1_byte, // narnia2
        3_byte,  1_byte, // narnia3
        4_byte,  1_byte, // narnia4
        5_byte,  1_byte, // narnia5
    };
    const size_t localNumberOfBlocks = 8;

    const std::vector<ReStore::SerializedBlocksDescriptor> blockDescriptors = {
        {0, 2, serializedData.data()},
        {4, 8, serializedData.data() + 4 * sizeof(World)}};
    auto sendBuffers = comm.copySerializedBlocksToSendBuffers(blockDescriptors, localNumberOfBlocks);

    // There are four blocks per range. Therefore, blocks 0 and 1 are stored on ranks 0, 3, and 6. Blocks 4, 5, 6, 7 are
    // stored on ranks 1, 4, and 7.
    std::vector<std::byte> expectedSendBuffer036 = {
        0_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        1_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 1
        0_byte,  0_byte,                                                 // earth
        10_byte, 1_byte,                                                 // narnia
    };

    std::vector<std::byte> expectedSendBuffer147 = {
        4_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 4
        7_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 7
        2_byte, 1_byte,                                                 // narnia2
        3_byte, 1_byte,                                                 // narnia3
        4_byte, 1_byte,                                                 // narnia4
        5_byte, 1_byte,                                                 // narnia5
    };

    ASSERT_EQ(sendBuffers.size(), 10);

    ASSERT_EQ(sendBuffers[0].size(), 20);
    ASSERT_EQ(sendBuffers[3].size(), 20);
    ASSERT_EQ(sendBuffers[6].size(), 20);

    ASSERT_EQ(sendBuffers[0], expectedSendBuffer036);
    ASSERT_EQ(sendBuffers[3], expectedSendBuffer036);
    ASSERT_EQ(sendBuffers[6], expectedSendBuffer036);

    ASSERT_EQ(sendBuffers[1].size(), 24);
    ASSERT_EQ(sendBuffers[4].size(), 24);
    ASSERT_EQ(sendBuffers[7].size(), 24);

    ASSERT_EQ(sendBuffers[1], expectedSendBuffer147);
    ASSERT_EQ(sendBuffers[4], expectedSendBuffer147);
    ASSERT_EQ(sendBuffers[7], expectedSendBuffer147);
}

TEST(BlockSubmissionTest, CopyAlreadySerializedBlocksToSendBuffers_multipleRangesSpanningMultiplePEs3) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;
    const size_t   numberOfPEs      = 10;
    const uint16_t replicationLevel = 3;
    const size_t   numberOfBlocks   = 20;

    struct World {
        bool    useMagic;
        uint8_t unicornCount;
    };

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(numberOfPEs));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(numberOfPEs));

    auto blockDistribution =
        std::make_shared<BlockDistribution>(numberOfPEs, numberOfBlocks, replicationLevel, mpiContext);

    ReStore::BlockSubmissionCommunication<World, MPIContextMock> comm(
        mpiContext, *blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, 2});

    std::vector<std::byte> serializedData = {
        0_byte,  0_byte, // earth
        10_byte, 1_byte, // narnia
        0_byte,  1_byte, // middle earth; not transmitted
        1_byte,  1_byte, // narnia1; not transmitted
        2_byte,  1_byte, // narnia2
        3_byte,  1_byte, // narnia3
        4_byte,  1_byte, // narnia4
        5_byte,  1_byte, // narnia5
    };
    const size_t localNumberOfBlocks = 8;

    const std::vector<ReStore::SerializedBlocksDescriptor> blockDescriptors = {
        {0, 2, serializedData.data()},                      // -> ranks 0, 3, 6
        {4, 8, serializedData.data() + 4 * sizeof(World)}}; // -> two blocks each: ranks 1, 4, 7 and ranks 2, 5, 8
    auto sendBuffers = comm.copySerializedBlocksToSendBuffers(blockDescriptors, localNumberOfBlocks);

    // There are two blocks per range. Therefore, blocks 0 and 1 are stored on ranks 0, 3, and 6. Blocks 4 and 5 are
    // stored on ranks 2, 5, and 8. Blocks 6 and 7 are stored on ranks 3, 6, and 9.
    std::vector<std::byte> expectedSendBuffer0 = {
        0_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        1_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 1
        0_byte,  0_byte,                                                 // earth
        10_byte, 1_byte,                                                 // narnia
    };

    std::vector<std::byte> expectedSendBuffer36 = {
        0_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        1_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 1
        0_byte,  0_byte,                                                 // earth
        10_byte, 1_byte,                                                 // narnia
        6_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 6
        7_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 7
        4_byte,  1_byte,                                                 // narnia4
        5_byte,  1_byte,                                                 // narnia5
    };

    std::vector<std::byte> expectedSendBuffer258 = {
        4_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 4
        5_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 7
        2_byte, 1_byte,                                                 // narnia2
        3_byte, 1_byte,                                                 // narnia3
    };

    std::vector<std::byte> expectedSendBuffer9 = {
        6_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 6
        7_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 7
        4_byte, 1_byte,                                                 // narnia4
        5_byte, 1_byte,                                                 // narnia5
    };

    std::vector<std::byte> expectedSendBuffer147 = {};

    ASSERT_EQ(sendBuffers.size(), 10);

    EXPECT_EQ(sendBuffers[0].size(), 20);
    EXPECT_EQ(sendBuffers[1].size(), 0);
    EXPECT_EQ(sendBuffers[2].size(), 20);
    EXPECT_EQ(sendBuffers[3].size(), 40);
    EXPECT_EQ(sendBuffers[4].size(), 0);
    EXPECT_EQ(sendBuffers[5].size(), 20);
    EXPECT_EQ(sendBuffers[6].size(), 40);
    EXPECT_EQ(sendBuffers[7].size(), 0);
    EXPECT_EQ(sendBuffers[8].size(), 20);
    EXPECT_EQ(sendBuffers[9].size(), 20);

    EXPECT_EQ(sendBuffers[0], expectedSendBuffer0);
    EXPECT_EQ(sendBuffers[1], expectedSendBuffer147);
    EXPECT_EQ(sendBuffers[2], expectedSendBuffer258);
    EXPECT_EQ(sendBuffers[3], expectedSendBuffer36);
    EXPECT_EQ(sendBuffers[4], expectedSendBuffer147);
    EXPECT_EQ(sendBuffers[5], expectedSendBuffer258);
    EXPECT_EQ(sendBuffers[6], expectedSendBuffer36);
    EXPECT_EQ(sendBuffers[7], expectedSendBuffer147);
    EXPECT_EQ(sendBuffers[8], expectedSendBuffer258);
    EXPECT_EQ(sendBuffers[9], expectedSendBuffer9);
}
TEST(BlockSubmissionTest, SerializeBlockForSubmission) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;

    struct World {
        bool    useMagic;
        uint8_t unicornCount;
    };

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(10));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(10));

    auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);

    ReStore::BlockSubmissionCommunication<World, MPIContextMock> comm(
        mpiContext, *blockDistribution, ReStore::OffsetModeDescriptor{OffsetMode::constant, 2});

    World              earth                 = {false, 0};
    World              narnia                = {true, 10};
    World              middleEarth           = {true, 0};
    std::vector<World> worlds                = {earth, narnia, middleEarth};
    const auto         numBlocks             = worlds.size();
    size_t             worldId               = 0;
    const auto         permutatorSeed        = 42;
    const auto         permutatorRangeLength = 1;

    auto sendBuffers = comm.serializeBlocksForTransmission(
        [](World world, ReStore::SerializedBlockStoreStream& stream) {
            stream << world.unicornCount;
            stream << world.useMagic;
        },
        [&worlds, &worldId]() -> std::optional<NextBlock<World>> {
            auto ret = worldId < worlds.size() ? std::make_optional(NextBlock<World>({worldId, worlds[worldId]}))
                                               : std::nullopt;
            ++worldId;
            return ret;
        },
        RangePermutation<IdentityPermutation>(numBlocks, permutatorRangeLength, permutatorSeed));

    // All these three blocks belong to range 0 and are therefore stored on ranks 0, 3 and 6
    // std::vector<std::byte> expectedSendBuffer = {
    //    0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte,  0_byte, // earth
    //    1_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 10_byte, 1_byte, // narnia
    //    2_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte,  1_byte  // middle earth
    //};
    std::vector<std::byte> expectedSendBuffer = {
        0_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // from block id 0
        2_byte,  0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, 0_byte, // to block id 2
        0_byte,  0_byte,                                                 // earth
        10_byte, 1_byte,                                                 // narnia
        0_byte,  1_byte                                                  // middle earth
    };

    ASSERT_EQ(sendBuffers.size(), 10);

    ASSERT_EQ(sendBuffers[0].size(), 22);
    ASSERT_EQ(sendBuffers[3].size(), 22);
    ASSERT_EQ(sendBuffers[6].size(), 22);

    ASSERT_EQ(sendBuffers[0], expectedSendBuffer);
    ASSERT_EQ(sendBuffers[3], expectedSendBuffer);
    ASSERT_EQ(sendBuffers[6], expectedSendBuffer);
}

TEST(BlockSubmissionTest, BlockIsAPointer) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::block_id_t;
    using ReStore::NextBlock;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::original_rank_t;
    using ReStoreMPI::RecvMessage;

    using BlockProxy = uint8_t*;

    const uint16_t       NUM_DIMENSIONS = 3;
    std::vector<uint8_t> data{10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42};

    NiceMock<MPIContextMock> mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    EXPECT_CALL(mpiContext, getOriginalSize()).WillRepeatedly(Return(10));
    EXPECT_CALL(mpiContext, getCurrentSize()).WillRepeatedly(Return(10));

    auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);

    ReStore::BlockSubmissionCommunication<BlockProxy, MPIContextMock> comm(
        mpiContext, *blockDistribution,
        ReStore::OffsetModeDescriptor{OffsetMode::constant, sizeof(uint8_t) * NUM_DIMENSIONS});

    assert(data.size() % NUM_DIMENSIONS == 0);
    const size_t numDataPointsLocal  = data.size() / NUM_DIMENSIONS;
    const size_t numDataPointsGlobal = numDataPointsLocal * 10ul;

    // Reserve memory for the Block Proxies (we are only pushing references around during
    // serializeBlocksForTransmission, the actual data is not copied).
    BlockProxy currentProxy = nullptr;

    const auto permutatorSeed        = 1337;
    const auto permutatorRangeLength = 1;

    unsigned counter     = 0;
    auto     sendBuffers = comm.serializeBlocksForTransmission(
        [](const BlockProxy& blockProxy, ReStore::SerializedBlockStoreStream& stream) {
            assert(blockProxy != nullptr);
            stream.writeBytes(reinterpret_cast<const std::byte*>(blockProxy), sizeof(uint8_t) * NUM_DIMENSIONS);
        },
        [&counter, data = data.data(), numDataPointsLocal,
         &currentProxy]() -> std::optional<ReStore::NextBlock<BlockProxy>> {
            assert(data != nullptr);
            std::optional<ReStore::NextBlock<BlockProxy>> nextBlock = std::nullopt;
            if (counter < numDataPointsLocal) {
                currentProxy = &data[counter * NUM_DIMENSIONS];
                nextBlock.emplace(counter, currentProxy);
                counter++; // We cannot put this in the above line, as we can't assume if the first argument of the
                           // pair is bound before or after the increment.
            }
            return nextBlock;
        },
        RangePermutation<IdentityPermutation>(numDataPointsGlobal, permutatorRangeLength, permutatorSeed));

    // All these three blocks belong to range 0 and are therefore stored on ranks 0, 3 and 6
    std::vector<std::byte> expectedSendBuffer = {
        0_byte,  0_byte,  0_byte,  0_byte,  0_byte,  0_byte,  0_byte,  0_byte, // from block id 0
        3_byte,  0_byte,  0_byte,  0_byte,  0_byte,  0_byte,  0_byte,  0_byte, // to block id 2
        10_byte, 11_byte, 12_byte, 20_byte, 21_byte, 22_byte, 30_byte, 31_byte, 32_byte, 40_byte, 41_byte, 42_byte};

    ASSERT_EQ(sendBuffers.size(), 10);

    ASSERT_EQ(sendBuffers[0].size(), 8 + 8 + sizeof(uint8_t) * data.size());
    ASSERT_EQ(sendBuffers[3].size(), 8 + 8 + sizeof(uint8_t) * data.size());
    ASSERT_EQ(sendBuffers[6].size(), 8 + 8 + sizeof(uint8_t) * data.size());

    ASSERT_EQ(sendBuffers[0], expectedSendBuffer);
    ASSERT_EQ(sendBuffers[3], expectedSendBuffer);
    ASSERT_EQ(sendBuffers[6], expectedSendBuffer);
}

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define __aicore__
#include "student_pipeline.h"
#undef __aicore__

namespace {

struct Token {
    uint32_t tile;
    char input;
};

struct Event {
    char kind;
    uint32_t tile;
};

void Require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class MockPipeline {
public:
    explicit MockPipeline(uint32_t tileCount) : tileCount_(tileCount)
    {
        copyInCount_.assign(tileCount, 0);
        computeCount_.assign(tileCount, 0);
        copyOutCount_.assign(tileCount, 0);
    }

    void CopyIn(uint32_t tile)
    {
        Require(tile < tileCount_, "CopyIn tile is out of range");
        Require(tile == nextCopyIn_, "CopyIn tiles must be ordered and unique");
        ++nextCopyIn_;
        ++copyInCount_[tile];
        xReady_.push_back(tile);
        yReady_.push_back(tile);
        ++xBuffersInUse_;
        ++yBuffersInUse_;
        maxXBuffersInUse_ = std::max(maxXBuffersInUse_, xBuffersInUse_);
        maxYBuffersInUse_ = std::max(maxYBuffersInUse_, yBuffersInUse_);
        Require(xBuffersInUse_ <= kStudentBufferNum,
            "input X uses more physical buffers than kStudentBufferNum");
        Require(yBuffersInUse_ <= kStudentBufferNum,
            "input Y uses more physical buffers than kStudentBufferNum");
        events_.push_back({'I', tile});
    }

    Token DeQueX()
    {
        Require(!xReady_.empty(), "DeQueX reads an empty queue");
        const uint32_t tile = xReady_.front();
        xReady_.pop_front();
        Require(!xHeld_, "two X tensors are held at once");
        xHeld_ = true;
        events_.push_back({'X', tile});
        return {tile, 'X'};
    }

    Token DeQueY()
    {
        Require(!yReady_.empty(), "DeQueY reads an empty queue");
        const uint32_t tile = yReady_.front();
        yReady_.pop_front();
        Require(!yHeld_, "two Y tensors are held at once");
        yHeld_ = true;
        events_.push_back({'Y', tile});
        return {tile, 'Y'};
    }

    void Compute(Token x, Token y)
    {
        Require(x.input == 'X' && y.input == 'Y', "Compute input type mismatch");
        Require(x.tile == y.tile, "Compute combines different tiles");
        Require(xHeld_ && yHeld_, "Compute requires two dequeued inputs");
        Require(x.tile < tileCount_, "Compute tile is out of range");
        ++computeCount_[x.tile];
        outputReady_.push_back(x.tile);
        xHeld_ = false;
        yHeld_ = false;
        --xBuffersInUse_;
        --yBuffersInUse_;
        events_.push_back({'C', x.tile});
    }

    void CopyOut(uint32_t tile)
    {
        Require(tile < tileCount_, "CopyOut tile is out of range");
        Require(!outputReady_.empty(), "CopyOut reads an empty queue");
        Require(outputReady_.front() == tile, "CopyOut order does not match Compute order");
        outputReady_.pop_front();
        ++copyOutCount_[tile];
        events_.push_back({'O', tile});
    }

    void Verify() const
    {
        Require(nextCopyIn_ == tileCount_, "not every tile was copied in");
        Require(xReady_.empty() && yReady_.empty(), "input queues were not drained");
        Require(outputReady_.empty(), "output queue was not drained");
        Require(!xHeld_ && !yHeld_, "an input tensor was not released");
        Require(xBuffersInUse_ == 0 && yBuffersInUse_ == 0,
            "physical input buffers were not released");

        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            Require(copyInCount_[tile] == 1, "each tile must be copied in exactly once");
            Require(computeCount_[tile] == 1, "each tile must be computed exactly once");
            Require(copyOutCount_[tile] == 1, "each tile must be copied out exactly once");
            Require(Position('I', tile) < Position('X', tile),
                "CopyIn must precede DeQueX");
            Require(Position('I', tile) < Position('Y', tile),
                "CopyIn must precede DeQueY");
            Require(Position('X', tile) < Position('C', tile) &&
                    Position('Y', tile) < Position('C', tile),
                "both DeQue operations must precede Compute");
            Require(Position('C', tile) < Position('O', tile),
                "Compute must precede CopyOut");
        }

        if (tileCount_ > 1) {
            Require(maxXBuffersInUse_ == 2 && maxYBuffersInUse_ == 2,
                "no observable input prefetch window was created");
            for (uint32_t tile = 0; tile + 1 < tileCount_; ++tile) {
                Require(Position('Y', tile) < Position('I', tile + 1) &&
                        Position('I', tile + 1) < Position('C', tile),
                    "next tile must be prefetched after DeQue and before current Compute");
            }
            for (uint32_t tile = 1; tile < tileCount_; ++tile) {
                Require(Position('O', tile - 1) < Position('C', tile),
                    "previous output must be drained before current Compute");
            }
        } else {
            Require(maxXBuffersInUse_ == 1 && maxYBuffersInUse_ == 1,
                "one-tile case should use one input slot");
        }
    }

    uint32_t MaxInputBuffers() const
    {
        return std::max(maxXBuffersInUse_, maxYBuffersInUse_);
    }

private:
    size_t Position(char kind, uint32_t tile) const
    {
        for (size_t index = 0; index < events_.size(); ++index) {
            if (events_[index].kind == kind && events_[index].tile == tile) {
                return index;
            }
        }
        throw std::runtime_error("required schedule event is missing");
    }

    uint32_t tileCount_;
    uint32_t nextCopyIn_ = 0;
    std::deque<uint32_t> xReady_;
    std::deque<uint32_t> yReady_;
    std::deque<uint32_t> outputReady_;
    bool xHeld_ = false;
    bool yHeld_ = false;
    int32_t xBuffersInUse_ = 0;
    int32_t yBuffersInUse_ = 0;
    int32_t maxXBuffersInUse_ = 0;
    int32_t maxYBuffersInUse_ = 0;
    std::vector<uint32_t> copyInCount_;
    std::vector<uint32_t> computeCount_;
    std::vector<uint32_t> copyOutCount_;
    std::vector<Event> events_;
};

}  // namespace

int main()
{
    try {
        Require(kStudentBufferNum == 2,
            "kStudentBufferNum must be 2 for the chapter practice");
        for (const uint32_t tileCount : {1U, 2U, 7U}) {
            MockPipeline pipeline(tileCount);
            StudentProcess(pipeline, tileCount);
            pipeline.Verify();
            std::cout << "SCHEDULE_PASS tile_count=" << tileCount
                      << " max_input_buffers=" << pipeline.MaxInputBuffers()
                      << std::endl;
        }
    } catch (const std::exception &error) {
        std::cerr << "SCHEDULE_FAIL " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

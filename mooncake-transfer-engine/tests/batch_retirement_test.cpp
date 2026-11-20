// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "batch_retirement_reaper.h"
#include "multi_transport.h"
#include "transfer_engine.h"
#include "transfer_engine_impl.h"
#include "transport/transport.h"

namespace mooncake {

// Exposes the transport layer of a TransferEngineImpl so the retirement
// contract can be tested without starting a metadata/transport stack.
class TransferEngineImplTestPeer {
   public:
    static void setMultiTransport(TransferEngineImpl& engine,
                                  std::shared_ptr<MultiTransport> transports) {
        engine.multi_transports_ = std::move(transports);
    }
};

// Hardware-free transport: it materialises the task/slice layout of a submitted
// batch and reports the transfer status the test asks for. No buffer is ever
// touched, so the retirement contract is testable on CPU-only runners.
class RetirementProbeTransport : public Transport {
   public:
    Status submitTransfer(
        BatchID batch_id,
        const std::vector<TransferRequest>& entries) override {
        auto& batch = toBatchDesc(batch_id);
        auto& task = batch.task_list.emplace_back();
        task.batch_id = batch_id;
        task.transport_ = this;
        for (const auto& entry : entries) {
            auto* slice = getSliceCache().allocate();
            slice->source_addr = entry.source;
            slice->length = entry.length;
            slice->opcode = entry.opcode;
            slice->target_id = entry.target_id;
            slice->status = Slice::PENDING;
            slice->ts = 0;
            slice->task = &task;
            task.slice_list.push_back(slice);
            task.total_bytes += entry.length;
        }
        task.slice_count = task.slice_list.size();
        return Status::OK();
    }

    Status getTransferStatus(BatchID, size_t, TransferStatus& status) override {
        ++query_count_;
        if (!query_status_.ok()) return query_status_;
        status.s = reported_status_;
        status.transferred_bytes = 0;
        return Status::OK();
    }

    void failQueries(Status status) { query_status_ = std::move(status); }

    void clearQueryFailure() { query_status_ = Status::OK(); }

    int queryCount() const { return query_count_; }

    int registerLocalMemory(void*, size_t, const std::string&, bool,
                            bool) override {
        return 0;
    }

    int unregisterLocalMemory(void*, bool) override { return 0; }

    int registerLocalMemoryBatch(const std::vector<BufferEntry>&,
                                 const std::string&) override {
        return 0;
    }

    int unregisterLocalMemoryBatch(const std::vector<void*>&) override {
        return 0;
    }

    const char* getName() const override { return "retirement-probe"; }

   private:
    Status query_status_;
    TransferStatusEnum reported_status_ = TransferStatusEnum::WAITING;
    int query_count_ = 0;
};

// Probe transport that reports an unsafe (give-up) terminal state.
class NonQuiescentProbeTransport : public RetirementProbeTransport {
   public:
    bool isBatchQuiescent(BatchID) override { return false; }
};

// Probe transport that can claim a bounded stop, so the abandon contract can
// be tested without hardware.
class AbandonProbeTransport : public RetirementProbeTransport {
   public:
    Status abandonBatch(BatchID, std::chrono::milliseconds,
                        AbandonOutcome& outcome) override {
        ++abandon_calls_;
        outcome = abandon_outcome_;
        return abandon_status_;
    }

    void setAbandonResult(Status status, AbandonOutcome outcome) {
        abandon_status_ = std::move(status);
        abandon_outcome_ = outcome;
    }

    int abandonCalls() const { return abandon_calls_; }

   private:
    Status abandon_status_;
    AbandonOutcome abandon_outcome_ = AbandonOutcome::kQuiescentRecoverable;
    int abandon_calls_ = 0;
};

namespace {

std::vector<Transport::TransferRequest> makeRequests(size_t count) {
    std::vector<Transport::TransferRequest> entries;
    for (size_t i = 0; i < count; ++i) {
        Transport::TransferRequest request{};
        request.opcode = Transport::TransferRequest::WRITE;
        request.source = nullptr;  // the probe never dereferences the buffer
        request.target_id = 0;
        request.target_offset = i * 64;
        request.length = 64;
        entries.push_back(request);
    }
    return entries;
}

// Owns a MultiTransport plus the probe transport and one submitted batch. The
// batch handle stays opaque to the test: only public transport calls are used.
class SubmittedBatchFixture {
   public:
    explicit SubmittedBatchFixture(size_t request_count)
        : local_server_name_("retirement-test") {
        transports_ =
            std::make_shared<MultiTransport>(nullptr, local_server_name_);
        probe_ = std::make_shared<RetirementProbeTransport>();
        batch_id_ = transports_->allocateBatchID(1);
        submitted_ =
            probe_->submitTransfer(batch_id_, makeRequests(request_count));
    }

    MultiTransport& transports() { return *transports_; }

    RetirementProbeTransport& probe() { return *probe_; }

    Transport::BatchID batchId() const { return batch_id_; }

    bool submitted() const { return submitted_.ok(); }

    Transport::BatchDesc& batch() { return Transport::toBatchDesc(batch_id_); }

    Transport::Slice* slice(size_t index) {
        return batch().task_list[0].slice_list[index];
    }

   private:
    std::string local_server_name_;
    std::shared_ptr<MultiTransport> transports_;
    std::shared_ptr<RetirementProbeTransport> probe_;
    Transport::BatchID batch_id_ = 0;
    Status submitted_;
};

}  // namespace

// The public facade must reach the retirement check through
// TransferEngineImpl; this cannot compile while the middle layer lacks the
// forwarding member that the wrapper calls.
TEST(BatchRetirementTest, PublicApiReportsRetiredEmptyBatch) {
    TransferEngine engine(false);
    if (engine.init(P2PHANDSHAKE, "127.0.0.1:12345") != 0) {
        // A transport such as AscendDirect needs a usable ACL context;
        // this exercise only makes sense where the facade can initialise.
        GTEST_SKIP() << "no usable transport in this environment";
    }

    const BatchID batch_id = engine.allocateBatchID(1);
    ASSERT_NE(batch_id, 0);

    bool retired = false;
    const Status status = engine.getBatchRetirementStatus(batch_id, retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired);

    EXPECT_TRUE(engine.freeBatchID(batch_id).ok());
}

// Same forwarding path without a metadata server: an empty batch is retired and
// can be released immediately.
TEST(BatchRetirementTest, ImplForwardsRetirementStatus) {
    std::string local_server_name = "retirement-impl-test";
    TransferEngineImpl engine(false);
    auto transports =
        std::make_shared<MultiTransport>(nullptr, local_server_name);
    TransferEngineImplTestPeer::setMultiTransport(engine, transports);

    const BatchID batch_id = engine.allocateBatchID(1);
    ASSERT_NE(batch_id, 0);

    bool retired = false;
    const Status status = engine.getBatchRetirementStatus(batch_id, retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired);

    EXPECT_TRUE(engine.freeBatchID(batch_id).ok());
}

// Borrowed buffers stay owner-held until every slice of every task has reached
// a terminal state; cleanup is refused in the meantime.
TEST(BatchRetirementTest, PendingSlicesKeepBatchBusy) {
    SubmittedBatchFixture fixture(2);
    ASSERT_TRUE(fixture.submitted());

    bool retired = true;
    Status status = fixture.transports().getBatchRetirementStatus(
        fixture.batchId(), retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_FALSE(retired);

    const Status busy = fixture.transports().freeBatchID(fixture.batchId());
    EXPECT_EQ(busy.code(), Status::Code::kBatchBusy) << busy.ToString();

    fixture.slice(0)->markSuccess();
    status = fixture.transports().getBatchRetirementStatus(fixture.batchId(),
                                                           retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_FALSE(retired) << "one slice is still in flight";

    fixture.slice(1)->markSuccess();
    status = fixture.transports().getBatchRetirementStatus(fixture.batchId(),
                                                           retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired);
    EXPECT_TRUE(fixture.transports().freeBatchID(fixture.batchId()).ok());
}

// A terminal slice count is not enough: a completion callback that has not
// returned still keeps the batch alive.
TEST(BatchRetirementTest, PendingCompletionCallbackBlocksRetirement) {
    SubmittedBatchFixture fixture(1);
    ASSERT_TRUE(fixture.submitted());
    fixture.slice(0)->markSuccess();

    fixture.batch().active_completion_callbacks.fetch_add(
        1, std::memory_order_acq_rel);

    bool retired = true;
    Status status = fixture.transports().getBatchRetirementStatus(
        fixture.batchId(), retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_FALSE(retired);

    const Status busy = fixture.transports().freeBatchID(fixture.batchId());
    EXPECT_EQ(busy.code(), Status::Code::kBatchBusy) << busy.ToString();

    fixture.batch().active_completion_callbacks.fetch_sub(
        1, std::memory_order_acq_rel);
    status = fixture.transports().getBatchRetirementStatus(fixture.batchId(),
                                                           retired);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired);
    EXPECT_TRUE(fixture.transports().freeBatchID(fixture.batchId()).ok());
}

// A failed transfer is still retired: the error is reported to the caller, yet
// the transport confirms that no access is in flight, so cleanup must succeed.
TEST(BatchRetirementTest, FailedTaskRetiresWithErrorStatus) {
    SubmittedBatchFixture fixture(2);
    ASSERT_TRUE(fixture.submitted());
    fixture.slice(0)->markFailed();
    fixture.slice(1)->markFailed();

    bool retired = false;
    const Status status = fixture.transports().getBatchRetirementStatus(
        fixture.batchId(), retired);
    EXPECT_FALSE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired)
        << "terminal failure with no in-flight access is retired";

    const Status freed = fixture.transports().freeBatchID(fixture.batchId());
    EXPECT_TRUE(freed.ok()) << freed.ToString();
}

// A query error alone never authorises buffer reuse; the batch stays owned.
TEST(BatchRetirementTest, QueryErrorDoesNotRetire) {
    SubmittedBatchFixture fixture(2);
    ASSERT_TRUE(fixture.submitted());
    fixture.probe().failQueries(Status::Socket("probe query failed"));

    bool retired = true;
    const Status status = fixture.transports().getBatchRetirementStatus(
        fixture.batchId(), retired);
    EXPECT_FALSE(status.ok()) << status.ToString();
    EXPECT_FALSE(retired);
    EXPECT_GT(fixture.probe().queryCount(), 0);

    const Status busy = fixture.transports().freeBatchID(fixture.batchId());
    EXPECT_EQ(busy.code(), Status::Code::kBatchBusy) << busy.ToString();

    // Recovery path: once the probe answers again and the task finishes, the
    // batch must become releasable.
    fixture.probe().clearQueryFailure();
    fixture.slice(0)->markSuccess();
    fixture.slice(1)->markSuccess();
    retired = false;
    EXPECT_TRUE(fixture.transports()
                    .getBatchRetirementStatus(fixture.batchId(), retired)
                    .ok());
    EXPECT_TRUE(retired);
    EXPECT_TRUE(fixture.transports().freeBatchID(fixture.batchId()).ok());
}

// Transport-level cleanup (used by transports that own their BatchDesc) applies
// the same callback gate.
TEST(BatchRetirementTest, TransportCleanupWaitsForCallbacks) {
    auto probe = std::make_shared<RetirementProbeTransport>();
    const BatchID batch_id = probe->allocateBatchID(1);
    auto& batch = Transport::toBatchDesc(batch_id);
    batch.active_completion_callbacks.fetch_add(1, std::memory_order_acq_rel);

    const Status busy = probe->freeBatchID(batch_id);
    EXPECT_EQ(busy.code(), Status::Code::kBatchBusy) << busy.ToString();

    batch.active_completion_callbacks.fetch_sub(1, std::memory_order_acq_rel);
    EXPECT_TRUE(probe->freeBatchID(batch_id).ok());
}

// A recoverable abandon proves that no access is in flight, so retirement is
// published and the caller may free the batch.
TEST(BatchRetirementTest, AbandonRecoverablePublishesTerminalState) {
    std::string local_server_name = "abandon-recoverable-test";
    MultiTransport transports(nullptr, local_server_name);
    auto probe = std::make_shared<AbandonProbeTransport>();
    const BatchID batch_id = transports.allocateBatchID(1);
    ASSERT_TRUE(probe->submitTransfer(batch_id, makeRequests(2)).ok());

    Transport::AbandonOutcome outcome =
        Transport::AbandonOutcome::kNotQuiescent;
    const Status status = transports.abandonBatch(
        batch_id, std::chrono::milliseconds(50), outcome);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(outcome, Transport::AbandonOutcome::kQuiescentRecoverable);
    EXPECT_EQ(probe->abandonCalls(), 1);

    // The abandoned task reports its failure status, but retirement proves the
    // buffers are safe and the batch can be freed.
    bool retired = false;
    const Status retired_status =
        transports.getBatchRetirementStatus(batch_id, retired);
    EXPECT_FALSE(retired_status.ok()) << retired_status.ToString();
    EXPECT_TRUE(retired);
    EXPECT_TRUE(transports.freeBatchID(batch_id).ok());
}

// A device fault stops the batch but proves nothing about buffer safety: the
// batch must stay pending so the caller shuts the engine down instead of
// releasing borrowed buffers.
TEST(BatchRetirementTest, AbandonDeviceFaultKeepsBatchPending) {
    std::string local_server_name = "abandon-device-fault-test";
    MultiTransport transports(nullptr, local_server_name);
    auto probe = std::make_shared<AbandonProbeTransport>();
    probe->setAbandonResult(Status::OK(),
                            Transport::AbandonOutcome::kQuiescentDeviceFault);
    const BatchID batch_id = transports.allocateBatchID(1);
    ASSERT_TRUE(probe->submitTransfer(batch_id, makeRequests(2)).ok());

    Transport::AbandonOutcome outcome =
        Transport::AbandonOutcome::kNotQuiescent;
    const Status status = transports.abandonBatch(
        batch_id, std::chrono::milliseconds(50), outcome);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(outcome, Transport::AbandonOutcome::kQuiescentDeviceFault);

    bool retired = true;
    EXPECT_TRUE(transports.getBatchRetirementStatus(batch_id, retired).ok());
    EXPECT_FALSE(retired);
    EXPECT_EQ(transports.freeBatchID(batch_id).code(),
              Status::Code::kBatchBusy);

    // Finish the fake transfer so the test does not leak the batch handle.
    auto& batch = Transport::toBatchDesc(batch_id);
    for (auto* slice : batch.task_list[0].slice_list) slice->markSuccess();
    EXPECT_TRUE(transports.getBatchRetirementStatus(batch_id, retired).ok());
    EXPECT_TRUE(retired);
    EXPECT_TRUE(transports.freeBatchID(batch_id).ok());
}

// A transport without the capability cannot prove safety; the generic layer
// must report kNotQuiescent and keep the batch pending.
TEST(BatchRetirementTest, AbandonUnsupportedTransportKeepsBatchPending) {
    std::string local_server_name = "abandon-unsupported-test";
    MultiTransport transports(nullptr, local_server_name);
    auto probe = std::make_shared<RetirementProbeTransport>();
    const BatchID batch_id = transports.allocateBatchID(1);
    ASSERT_TRUE(probe->submitTransfer(batch_id, makeRequests(2)).ok());

    Transport::AbandonOutcome outcome =
        Transport::AbandonOutcome::kQuiescentRecoverable;
    const Status status = transports.abandonBatch(
        batch_id, std::chrono::milliseconds(50), outcome);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(outcome, Transport::AbandonOutcome::kNotQuiescent);

    bool retired = true;
    EXPECT_TRUE(transports.getBatchRetirementStatus(batch_id, retired).ok());
    EXPECT_FALSE(retired);

    auto& batch = Transport::toBatchDesc(batch_id);
    for (auto* slice : batch.task_list[0].slice_list) slice->markSuccess();
    EXPECT_TRUE(transports.getBatchRetirementStatus(batch_id, retired).ok());
    EXPECT_TRUE(transports.freeBatchID(batch_id).ok());
}

// A give-up terminal state retires the batch but must not be advertised as
// safe to release.
TEST(BatchRetirementTest, NonQuiescentRetirementIsReported) {
    std::string local_server_name = "non-quiescent-test";
    MultiTransport transports(nullptr, local_server_name);
    auto probe = std::make_shared<NonQuiescentProbeTransport>();
    const BatchID batch_id = transports.allocateBatchID(1);
    ASSERT_TRUE(probe->submitTransfer(batch_id, makeRequests(2)).ok());
    auto& batch = Transport::toBatchDesc(batch_id);
    for (auto* slice : batch.task_list[0].slice_list) slice->markSuccess();

    bool retired = false;
    bool quiescent = true;
    const Status status =
        transports.getBatchRetirementStatus(batch_id, retired, quiescent);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired);
    EXPECT_FALSE(quiescent);
    EXPECT_TRUE(transports.freeBatchID(batch_id).ok());
}

// A confirmed stop is reported as quiescent.
TEST(BatchRetirementTest, QuiescentRetirementIsReported) {
    std::string local_server_name = "quiescent-test";
    MultiTransport transports(nullptr, local_server_name);
    auto probe = std::make_shared<RetirementProbeTransport>();
    const BatchID batch_id = transports.allocateBatchID(1);
    ASSERT_TRUE(probe->submitTransfer(batch_id, makeRequests(2)).ok());
    auto& batch = Transport::toBatchDesc(batch_id);
    for (auto* slice : batch.task_list[0].slice_list) slice->markSuccess();

    bool retired = false;
    bool quiescent = false;
    const Status status =
        transports.getBatchRetirementStatus(batch_id, retired, quiescent);
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(retired);
    EXPECT_TRUE(quiescent);
    EXPECT_TRUE(transports.freeBatchID(batch_id).ok());
}

// The asynchronous reaper must release buffers only after the transport
// proved quiescence, and must not invoke the callback before that point.
TEST(BatchRetirementTest, AsyncReleasesOnlyAfterQuiescentRetirement) {
    std::atomic<int> queries{0};
    std::atomic<int> frees{0};
    std::atomic<bool> released{false};
    std::promise<BatchRetirementState> done;
    BatchRetirementOptions options;
    options.abandon_delay = std::chrono::seconds(10);
    options.deadline = std::chrono::seconds(5);

    BatchRetirementReaper reaper(
        [&](BatchID, bool& retired, bool& quiescent) {
            retired = ++queries >= 3;
            quiescent = true;
            return Status::OK();
        },
        [](BatchID, std::chrono::milliseconds,
           Transport::AbandonOutcome& outcome) {
            outcome = Transport::AbandonOutcome::kNotQuiescent;
            return Status::OK();
        },
        [&](BatchID) {
            ++frees;
            return Status::OK();
        });

    ASSERT_TRUE(reaper
                    .enqueue(7, options,
                             [&](BatchID, const Status& status,
                                 BatchRetirementState state) {
                                 EXPECT_TRUE(status.ok()) << status.ToString();
                                 released = true;
                                 done.set_value(state);
                             })
                    .ok());
    EXPECT_FALSE(released.load());
    auto future = done.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(future.get(), BatchRetirementState::kSafeToRelease);
    EXPECT_EQ(frees.load(), 1);
}

// Retirement alone is enough to free the batch. Missing quiescence proof is
// still reported as kUnrecoverable, but it no longer leaves the descriptor
// allocated.
TEST(BatchRetirementTest, AsyncFreesRetiredBatchWithoutQuiescence) {
    std::atomic<int> frees{0};
    std::promise<BatchRetirementState> done;
    BatchRetirementOptions options;
    options.abandon_delay = std::chrono::seconds(10);
    options.deadline = std::chrono::seconds(5);

    BatchRetirementReaper reaper(
        [](BatchID, bool& retired, bool& quiescent) {
            retired = true;
            quiescent = false;
            return Status::OK();
        },
        [](BatchID, std::chrono::milliseconds,
           Transport::AbandonOutcome& outcome) {
            outcome = Transport::AbandonOutcome::kNotQuiescent;
            return Status::OK();
        },
        [&](BatchID) {
            ++frees;
            return Status::OK();
        });

    ASSERT_TRUE(reaper
                    .enqueue(10, options,
                             [&](BatchID, const Status& status,
                                 BatchRetirementState state) {
                                 EXPECT_FALSE(status.ok());
                                 done.set_value(state);
                             })
                    .ok());
    auto future = done.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(future.get(), BatchRetirementState::kUnrecoverable);
    EXPECT_EQ(frees.load(), 1);
}

// A caller that already saw a failure asks for an immediate abandon; a
// recoverable outcome must still end in a safe release.
TEST(BatchRetirementTest, AsyncAbandonsImmediatelyAndReleases) {
    std::atomic<bool> abandoned{false};
    std::atomic<int> abandons{0};
    std::atomic<int> frees{0};
    std::promise<BatchRetirementState> done;
    BatchRetirementOptions options;
    options.abandon_delay = std::chrono::milliseconds(0);
    options.abandon_timeout = std::chrono::milliseconds(10);
    options.deadline = std::chrono::seconds(2);

    BatchRetirementReaper reaper(
        [&](BatchID, bool& retired, bool& quiescent) {
            retired = abandoned.load();
            quiescent = true;
            return Status::OK();
        },
        [&](BatchID, std::chrono::milliseconds,
            Transport::AbandonOutcome& outcome) {
            ++abandons;
            abandoned = true;
            outcome = Transport::AbandonOutcome::kQuiescentRecoverable;
            return Status::OK();
        },
        [&](BatchID) {
            ++frees;
            return Status::OK();
        });

    ASSERT_TRUE(
        reaper
            .enqueue(8, options,
                     [&](BatchID, const Status&, BatchRetirementState state) {
                         done.set_value(state);
                     })
            .ok());
    auto future = done.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(future.get(), BatchRetirementState::kSafeToRelease);
    EXPECT_EQ(abandons.load(), 1);
    EXPECT_EQ(frees.load(), 1);
}

// A device fault means the transport cannot stop the batch: the callback must
// report kUnrecoverable and nothing may be freed.
TEST(BatchRetirementTest, AsyncDeviceFaultDoesNotRelease) {
    std::atomic<int> frees{0};
    std::promise<BatchRetirementState> done;
    BatchRetirementOptions options;
    options.abandon_delay = std::chrono::milliseconds(0);
    options.abandon_timeout = std::chrono::milliseconds(10);
    options.deadline = std::chrono::seconds(5);

    BatchRetirementReaper reaper(
        [](BatchID, bool& retired, bool& quiescent) {
            retired = false;
            quiescent = true;
            return Status::OK();
        },
        [](BatchID, std::chrono::milliseconds,
           Transport::AbandonOutcome& outcome) {
            outcome = Transport::AbandonOutcome::kQuiescentDeviceFault;
            return Status::OK();
        },
        [&](BatchID) {
            ++frees;
            return Status::OK();
        });

    ASSERT_TRUE(
        reaper
            .enqueue(9, options,
                     [&](BatchID, const Status&, BatchRetirementState state) {
                         done.set_value(state);
                     })
            .ok());
    auto future = done.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(future.get(), BatchRetirementState::kUnrecoverable);
    EXPECT_EQ(frees.load(), 0);
}

// Without a quiescence proof the deadline turns into a fail-closed answer
// instead of an unbounded wait.
TEST(BatchRetirementTest, AsyncDeadlineIsUnrecoverable) {
    std::atomic<int> frees{0};
    std::promise<BatchRetirementState> done;
    BatchRetirementOptions options;
    options.abandon_delay = std::chrono::milliseconds(0);
    options.abandon_timeout = std::chrono::milliseconds(10);
    options.deadline = std::chrono::milliseconds(100);

    BatchRetirementReaper reaper(
        [](BatchID, bool& retired, bool& quiescent) {
            retired = false;
            quiescent = true;
            return Status::OK();
        },
        [](BatchID, std::chrono::milliseconds,
           Transport::AbandonOutcome& outcome) {
            outcome = Transport::AbandonOutcome::kNotQuiescent;
            return Status::OK();
        },
        [&](BatchID) {
            ++frees;
            return Status::OK();
        });

    ASSERT_TRUE(
        reaper
            .enqueue(10, options,
                     [&](BatchID, const Status&, BatchRetirementState state) {
                         done.set_value(state);
                     })
            .ok());
    auto future = done.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(future.get(), BatchRetirementState::kUnrecoverable);
    EXPECT_EQ(frees.load(), 0);
}

// A quarantined batch must not delay the retirement of later batches: the
// submission path returns as soon as the batch is enqueued.
TEST(BatchRetirementTest, AsyncQuarantinedBatchDoesNotBlockLaterBatches) {
    std::promise<BatchID> done;
    std::atomic<bool> stuck_done{false};
    BatchRetirementOptions options;
    options.abandon_delay = std::chrono::seconds(10);
    options.deadline = std::chrono::seconds(10);

    BatchRetirementReaper reaper(
        [](BatchID batch_id, bool& retired, bool& quiescent) {
            retired = batch_id != 11;
            quiescent = true;
            return Status::OK();
        },
        [](BatchID, std::chrono::milliseconds,
           Transport::AbandonOutcome& outcome) {
            outcome = Transport::AbandonOutcome::kNotQuiescent;
            return Status::OK();
        },
        [](BatchID) { return Status::OK(); });

    ASSERT_TRUE(reaper
                    .enqueue(11, options,
                             [&](BatchID, const Status&, BatchRetirementState) {
                                 stuck_done = true;
                             })
                    .ok());
    ASSERT_TRUE(
        reaper
            .enqueue(12, options,
                     [&](BatchID batch_id, const Status&,
                         BatchRetirementState) { done.set_value(batch_id); })
            .ok());
    auto future = done.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(future.get(), 12);
    EXPECT_FALSE(stuck_done.load());
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

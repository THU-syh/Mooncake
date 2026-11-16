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

#ifndef MOONCAKE_BATCH_RETIREMENT_REAPER_H
#define MOONCAKE_BATCH_RETIREMENT_REAPER_H

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "batch_retirement.h"
#include "transport/transport.h"

namespace mooncake {

// Drives quarantined batches to a terminal state on a dedicated thread so the
// caller that observed a transfer failure can return immediately. The transfer
// submission thread never waits here; only this reaper does.
class BatchRetirementReaper {
   public:
    using QueryRetirementFn = std::function<Status(BatchID, bool&, bool&)>;
    using AbandonFn = std::function<Status(BatchID, std::chrono::milliseconds,
                                           Transport::AbandonOutcome&)>;
    using FreeBatchFn = std::function<Status(BatchID)>;

    BatchRetirementReaper(QueryRetirementFn query_retirement, AbandonFn abandon,
                          FreeBatchFn free_batch);
    ~BatchRetirementReaper();

    BatchRetirementReaper(const BatchRetirementReaper&) = delete;
    BatchRetirementReaper& operator=(const BatchRetirementReaper&) = delete;

    // Enqueues a batch and returns without waiting. The callback runs on the
    // reaper thread exactly once and must not block.
    Status enqueue(BatchID batch_id, const BatchRetirementOptions& options,
                   BatchRetirementCallback callback);

    // Stops the worker and drops pending callbacks without invoking them.
    // Used when the owning engine is being torn down.
    void stop();

   private:
    struct PendingBatch {
        BatchID batch_id;
        BatchRetirementOptions options;
        BatchRetirementCallback callback;
        std::chrono::steady_clock::time_point started_at;
        Status first_error{Status::OK()};
        bool abandon_attempted{false};
        std::size_t cleanup_failures{0};
    };

    // Terminal action collected under the lock and invoked after unlocking.
    struct Completion {
        BatchRetirementCallback callback;
        BatchID batch_id;
        Status status;
        BatchRetirementState state;
    };

    void run();
    // Advances one batch. Either updates it in place and returns false, or
    // fills `completion` and returns true.
    bool advance(PendingBatch& batch, Completion& completion);
    static constexpr std::size_t kCleanupRetryAttempts = 3;

    QueryRetirementFn query_retirement_;
    AbandonFn abandon_;
    FreeBatchFn free_batch_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<PendingBatch> pending_;
    bool stopping_{false};
    std::thread worker_;
};

}  // namespace mooncake

#endif  // MOONCAKE_BATCH_RETIREMENT_REAPER_H

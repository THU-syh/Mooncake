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

#include <mutex>
#include <utility>

#include "batch_retirement_reaper.h"
#include "transfer_engine.h"

namespace mooncake {

namespace {

// The reaper is created lazily on the first asynchronous retirement. A
// function-local mutex keeps TransferEngine movable: the mutex is not a member
// and therefore never participates in the move.
std::mutex& retirement_reaper_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

void TransferEngine::stop_retirement_reaper() {
    std::shared_ptr<BatchRetirementReaper> reaper;
    {
        std::lock_guard<std::mutex> lock(retirement_reaper_mutex());
        reaper = std::move(retirement_reaper_);
    }
    // Join outside the lock: a callback that races with teardown must not be
    // able to deadlock against this mutex.
    if (reaper) {
        reaper->stop();
    }
}

Status TransferEngine::retireBatchAsync(BatchID batch_id,
                                        const BatchRetirementOptions& options,
                                        BatchRetirementCallback callback) {
    if (!callback) {
        return Status::InvalidArgument("retirement callback is empty");
    }
    std::lock_guard<std::mutex> lock(retirement_reaper_mutex());
    if (!retirement_reaper_) {
        retirement_reaper_ = std::make_shared<BatchRetirementReaper>(
            [this](BatchID id, bool& retired, bool& quiescent) {
                return getBatchRetirementStatus(id, retired, quiescent);
            },
            [this](BatchID id, std::chrono::milliseconds timeout,
                   Transport::AbandonOutcome& outcome) {
                return abandonBatch(id, timeout, outcome);
            },
            [this](BatchID id) { return freeBatchID(id); });
    }
    return retirement_reaper_->enqueue(batch_id, options, std::move(callback));
}

}  // namespace mooncake

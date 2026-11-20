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

#include "batch_retirement_reaper.h"

#include <algorithm>

namespace mooncake {

namespace {
constexpr std::chrono::milliseconds kMinPollInterval{1};
}  // namespace

BatchRetirementReaper::BatchRetirementReaper(QueryRetirementFn query_retirement,
                                             AbandonFn abandon,
                                             FreeBatchFn free_batch)
    : query_retirement_(std::move(query_retirement)),
      abandon_(std::move(abandon)),
      free_batch_(std::move(free_batch)),
      worker_([this]() { run(); }) {}

BatchRetirementReaper::~BatchRetirementReaper() { stop(); }

Status BatchRetirementReaper::enqueue(BatchID batch_id,
                                      const BatchRetirementOptions& options,
                                      BatchRetirementCallback callback) {
    if (!callback) {
        return Status::InvalidArgument("retirement callback is empty");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return Status::Context("retirement reaper is stopping");
        }
        PendingBatch batch;
        batch.batch_id = batch_id;
        batch.options = options;
        batch.callback = std::move(callback);
        batch.started_at = std::chrono::steady_clock::now();
        pending_.emplace_back(std::move(batch));
    }
    cv_.notify_one();
    return Status::OK();
}

void BatchRetirementReaper::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return;
        }
        // Pending callbacks are dropped on purpose: the owning engine is going
        // away, so nobody may release buffers through this reaper any more.
        stopping_ = true;
        pending_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void BatchRetirementReaper::run() {
    for (;;) {
        std::vector<Completion> completions;
        std::chrono::milliseconds wait_interval = kMinPollInterval;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (pending_.empty()) {
                cv_.wait(lock,
                         [this]() { return stopping_ || !pending_.empty(); });
            }
            if (stopping_) {
                break;
            }
            for (auto it = pending_.begin(); it != pending_.end();) {
                Completion completion;
                if (advance(*it, completion)) {
                    completions.emplace_back(std::move(completion));
                    it = pending_.erase(it);
                    continue;
                }
                wait_interval = std::min(
                    wait_interval,
                    std::max(it->options.poll_interval, kMinPollInterval));
                ++it;
            }
            if (!pending_.empty()) {
                cv_.wait_for(lock, wait_interval);
            }
        }
        for (auto& completion : completions) {
            completion.callback(completion.batch_id, completion.status,
                                completion.state);
        }
    }
}

bool BatchRetirementReaper::advance(PendingBatch& batch,
                                    Completion& completion) {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - batch.started_at;

    if (elapsed >= batch.options.deadline) {
        completion.batch_id = batch.batch_id;
        completion.callback = std::move(batch.callback);
        completion.status = batch.first_error.ok()
                                ? Status::Context(
                                      "Mooncake batch retirement deadline "
                                      "exceeded")
                                : batch.first_error;
        completion.state = BatchRetirementState::kUnrecoverable;
        return true;
    }

    if (!batch.abandon_attempted && elapsed >= batch.options.abandon_delay) {
        batch.abandon_attempted = true;
        Transport::AbandonOutcome outcome =
            Transport::AbandonOutcome::kNotQuiescent;
        const Status abandon_status =
            abandon_(batch.batch_id, batch.options.abandon_timeout, outcome);
        if (abandon_status.ok() &&
            outcome == Transport::AbandonOutcome::kQuiescentDeviceFault) {
            completion.batch_id = batch.batch_id;
            completion.callback = std::move(batch.callback);
            completion.status = abandon_status;
            completion.state = BatchRetirementState::kUnrecoverable;
            return true;
        }
    }

    bool retired = false;
    bool quiescent = true;
    const Status query_status =
        query_retirement_(batch.batch_id, retired, quiescent);
    if (batch.first_error.ok() && !query_status.ok()) {
        batch.first_error = query_status;
    }

    if (!retired) {
        return false;
    }

    // Retirement proves every slice is terminal, so the batch may be freed
    // even when the transport could not prove the hardware stopped. The state
    // still reports the missing proof so a caller that cannot tolerate a late
    // write can choose to restart; the owning policy decides what to do.
    const bool unsafe = !quiescent;
    if (unsafe && batch.first_error.ok()) {
        batch.first_error =
            Status::Context("Mooncake batch retired without quiescence proof");
    }

    const Status freed = free_batch_(batch.batch_id);
    if (freed.IsBatchBusy()) {
        return false;
    }
    if (!freed.ok()) {
        if (++batch.cleanup_failures < kCleanupRetryAttempts) {
            return false;
        }
        // Retirement made the batch freeable; if the descriptor still cannot
        // be cleaned up, report the cleanup error rather than retrying
        // forever. The quiescence state is preserved in the completion.
        completion.batch_id = batch.batch_id;
        completion.callback = std::move(batch.callback);
        completion.status = freed;
        completion.state = unsafe ? BatchRetirementState::kUnrecoverable
                                  : BatchRetirementState::kSafeToRelease;
        return true;
    }

    completion.batch_id = batch.batch_id;
    completion.callback = std::move(batch.callback);
    completion.status = batch.first_error;
    completion.state = unsafe ? BatchRetirementState::kUnrecoverable
                              : BatchRetirementState::kSafeToRelease;
    return true;
}

}  // namespace mooncake

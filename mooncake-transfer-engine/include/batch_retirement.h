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

#ifndef MOONCAKE_BATCH_RETIREMENT_H
#define MOONCAKE_BATCH_RETIREMENT_H

#include <chrono>
#include <functional>

#include "transport/transport.h"

namespace mooncake {

using BatchID = Transport::BatchID;

// Options for an asynchronous retirement. `deadline` bounds how long a batch
// may stay quarantined; when it expires the callback reports kUnrecoverable so
// the owner can fail closed instead of releasing buffers it cannot trust.
struct BatchRetirementOptions {
    // Time to wait before the first abandon attempt. Zero means "abandon now",
    // which is what a caller that already observed a transfer failure wants.
    std::chrono::milliseconds abandon_delay{0};
    // Bound for a single abandon attempt.
    std::chrono::milliseconds abandon_timeout{1000};
    // Bound for the whole retirement; exceeding it reports kUnrecoverable.
    std::chrono::milliseconds deadline{30000};
    // Poll interval while the batch is pending.
    std::chrono::milliseconds poll_interval{1};
};

enum class BatchRetirementState {
    // The transport proved it stopped touching the buffers; they may be
    // released, and the batch ID has been freed by the reaper.
    kSafeToRelease,
    // No quiescence proof: the batch was freed when it retired, otherwise the
    // descriptor remains allocated. The owner decides whether to fail the
    // request or restart; reused buffers must not be trusted to stay untouched.
    kUnrecoverable,
};

using BatchRetirementCallback =
    std::function<void(BatchID, const Status&, BatchRetirementState)>;

}  // namespace mooncake

#endif  // MOONCAKE_BATCH_RETIREMENT_H

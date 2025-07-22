/*
 *
 * See the blog post for details.
 *
 * Compile with:
 *
 *	       g++ res/futex/benchmark.cc -o bm -std=c++20 -lfmt -O3 && ./bm
 *
 */

#include <stdint.h>
#include <sys/time.h>
#include <linux/futex.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mutex>
#include <array>
#include <thread>
#include <shared_mutex>

#include <fmt/core.h>
#include <chrono>

#include <sys/resource.h>

// #define dbgPrint(...) fmt::print( __VA_ARGS__ );
#define dbgPrint(...) {}

namespace {

	rusage get_rusage() {
		rusage usage;
		// getrusage( RUSAGE_THREAD, &usage );
		getrusage( RUSAGE_SELF, &usage ); // whole process!
		return usage;
	}
	rusage print_rusage() {
		auto usage = get_rusage();
		fmt::print(" - rusage:\n");
		// fmt::print("       - ru_utime:    {}\n", usage.ru_utime);
		// fmt::print("       - ru_stime:    {}\n", usage.ru_stime);
		fmt::print("       - ru_maxrss:   {}\n", usage.ru_maxrss);
		fmt::print("       - ru_ixrss:    {}\n", usage.ru_ixrss);
		fmt::print("       - ru_idrss:    {}\n", usage.ru_idrss);
		fmt::print("       - ru_isrss:    {}\n", usage.ru_isrss);
		fmt::print("       - ru_minflt:   {}\n", usage.ru_minflt);
		fmt::print("       - ru_majflt:   {}\n", usage.ru_majflt);
		fmt::print("       - ru_nswap:    {}\n", usage.ru_nswap);
		fmt::print("       - ru_inblock:  {}\n", usage.ru_inblock);
		fmt::print("       - ru_oublock:  {}\n", usage.ru_oublock);
		fmt::print("       - ru_msgsnd:   {}\n", usage.ru_msgsnd);
		fmt::print("       - ru_msgrcv:   {}\n", usage.ru_msgrcv);
		fmt::print("       - ru_nsignals: {}\n", usage.ru_nsignals);
		fmt::print("       - ru_nvcsw:    {}\n", usage.ru_nvcsw); // Voluntary context switches
		fmt::print("       - ru_nivcsw:   {}\n", usage.ru_nivcsw); // Involuntary context switches
		return usage;
	}
	struct PrintContextSwitchesGuard {
		rusage usage0;
		inline PrintContextSwitchesGuard() {
			usage0 = get_rusage();
		}
		inline ~PrintContextSwitchesGuard() {
			rusage usage1 = get_rusage();
			fmt::print(" - Context switches (v {}) (iv {})\n",
					usage1.ru_nvcsw - usage0.ru_nvcsw,
					usage1.ru_nivcsw - usage0.ru_nivcsw);
		}
	};

	inline int64_t getMicros() {
		// timespec start;
		// clock_gettime(CLOCK_REALTIME, &start);
		// return start.tv_sec * 1'000'000 + start.tv_nsec / 1'000;
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	}

	long futex(uint32_t* uaddr, int futex_op, uint32_t val) {
		return syscall(SYS_futex, uaddr, futex_op, val);
	}
	long futex_wait_masked(uint32_t* uaddr, uint32_t val, uint32_t mask) {
		return syscall(SYS_futex, uaddr, FUTEX_WAIT_BITSET, val, 0, 0, mask);
	}
	long futex_wake_masked(uint32_t* uaddr, uint32_t val, uint32_t mask) {
		return syscall(SYS_futex, uaddr, FUTEX_WAKE_BITSET, val, 0, 0, mask);
	}

	//
	// The *CvHeader "concept" is implemented here by FutexCvHeader and PthreadCvHeader.
	//

	struct FutexCvHeader {
		uint32_t value;

		inline int  wake(uint32_t mask) {
			dbgPrint(" - wake {:>032b}\n", mask);
			uint32_t old = value;
			value = old;
			// return futex_wake_masked(&value, old+1, mask);
			return futex_wake_masked(&value, INT_MAX, mask);
		}

		long wait(uint32_t mask) {
			while (true) {
				uint32_t expected = value;
				auto stat = futex_wait_masked(&value, expected, mask);
				if (stat == EAGAIN) {
					// Does not seem to happen on x86_64
					fmt::print(" - futex wait EAGAIN\n");
					continue;
				}
				return 0;

			}
			return 0;
		}

		void initialize() { value = 0; }
	};

	struct PthreadCvHeader {
		pthread_cond_t cv;
		pthread_mutex_t mtx;

		inline int wake(uint32_t mask) {
			pthread_cond_broadcast(&cv);
			return 0;
		}

		long wait(uint32_t mask) {
			pthread_mutex_lock(&mtx);
			while (true) {
				pthread_cond_wait(&cv, &mtx);
				pthread_mutex_unlock(&mtx);
				return 0;
			}
			return 0;
		}

		void initialize() {
			pthread_cond_init(&cv, nullptr);
			pthread_mutex_init(&mtx, nullptr);
		}

		~PthreadCvHeader() {
			pthread_cond_broadcast(&cv);
			pthread_cond_destroy(&cv);
			pthread_mutex_destroy(&mtx);
		}
	};



	//
	// The communication bus types.
	// A Domain holds multiple channels.
	// A Channel holds a message buffer (would be dynamic allocation in real use case), a sequence counter, and a shared_mutex (pthread_rwlock_t).
	// Finally, Writers and Readers work on the Channels and require exclusive or shared locks respectively.
	//


	static constexpr int Writers = 8;
	static constexpr int Readers = 32;
	// static constexpr int Readers = 8;

	constexpr int MessageSize = 64;
	struct Message {
		int64_t t;
		uint8_t extraBytes[MessageSize-8];
	};

	struct Channel {
		std::shared_mutex mtx;
		Message message;
		std::atomic<int32_t> seq;
		inline Channel() {
			seq = 0;
		}
	};

	template <class Cv>
	struct Domain {
		Cv cv;
		std::array<Channel,Writers> channels;
		bool stop = false;
	};

	template <class Cv>
	struct Writer {
		Domain<Cv> &domain;

		std::thread thread;
		uint32_t channel;
		int64_t nwrite=0;

		inline Writer(Domain<Cv>& domain_, uint32_t channel_) : domain(domain_), channel(channel_) {
			thread = std::thread(&Writer<Cv>::loop, this);
		}
		inline ~Writer() {
			thread.join();
		}
		inline void loop() {
			while (!domain.stop) {
				// usleep(100'000 + channel * 5'000);
				if (channel < 4)
					usleep(2'000 + channel * 0);
				else
					usleep(20'000 + channel * 1000);
				// usleep(2'100'000 + channel * 105'000);

				{
					std::unique_lock<std::shared_mutex> lck(domain.channels[channel].mtx);
					domain.channels[channel].message = Message{.t=getMicros()};
					domain.channels[channel].seq++;
					nwrite++;
				}

				dbgPrint(" - notified channel {}\n", channel);
				domain.cv.wake(1 << channel);
			}
		}
	};

	template <class Cv>
	struct Reader {
		Domain<Cv> &domain;

		int32_t readerId;
		std::thread thread;
		uint32_t channelMask;

		int64_t sumLatency=0;
		int64_t nread=0;
		std::array<int32_t,Readers> lastSeq;

		inline Reader(Domain<Cv>& domain_, uint32_t readerId_, uint32_t channelMask_) : domain(domain_), readerId(readerId_), channelMask(channelMask_) {
			dbgPrint(" - register reader {:>2d} mask {:>032b}\n", readerId, channelMask);
			thread = std::thread(&Reader<Cv>::loop, this);
			for (int i=0; i<Readers; i++) lastSeq[i] = 0;
		}
		inline ~Reader() {
			thread.join();
		}
		inline void loop() {
			while (!domain.stop) {

				auto stat = domain.cv.wait(channelMask);
				dbgPrint(" - awake reader {:>2d} mask {:>032b} stat {}\n", readerId, channelMask, stat);
				if (domain.stop) break;
				for (uint32_t i=0; i<Writers; i++)
					if (channelMask & (1 << i)) {
						auto newSeq = domain.channels[i].seq.load();
						dbgPrint(" - newSeq {} vs lastSeq[i] {}\n", newSeq, lastSeq[i]);
						if (lastSeq[i] < newSeq) {
							int64_t msg_time;
							{
								std::shared_lock<std::shared_mutex> lck(domain.channels[i].mtx);
								msg_time = domain.channels[i].message.t;
							}
							auto lat = getMicros() - msg_time;
							sumLatency += lat;
							nread++;
							lastSeq[i] = newSeq;
						}
					}


			}
		}
	};

	//
	// Create 8 Writers and 32 Readers.
	// Each Reader listens to 4 Writers.
	// I also wire each Reader to read from Channel1, to simulate a busy channel.
	//
	//


	template <class Cv>
	struct Benchmark {
		std::array<Writer<Cv>*,Writers> writers;
		std::array<Reader<Cv>*,Readers> readers;

		inline void run() {

			Domain<Cv> domain;
			domain.cv.initialize();

			// Make _all_ readers listen to channel 1
			for (uint32_t i=0; i<Readers; i++) readers[i] = new Reader{domain, i, 1u
				| (1<<(i/(Readers/Writers)))
				| (1<<(i/(Readers/Writers)+1))
				| (1<<(i/(Readers/Writers)+2))
				| (1<<(i/(Readers/Writers)+3))
			};
			// for (int i=0; i<Readers; i++) readers[i] = new Reader{domain, i, (1<<(i/(Readers/Writers))) | (1<<(i/(Readers/Writers)+1))};
			for (uint32_t i=0; i<Writers; i++) writers[i] = new Writer{domain, i};

			sleep(5);

			domain.stop = true;
			domain.cv.wake(~0u);
			usleep(20'000);
			domain.cv.wake(~0u);
			usleep(10'000);

			int64_t latencies = 0;
			int64_t nr = 0, nw = 0;

			for (int i=0; i<Writers; i++) nw += writers[i]->nwrite;
			for (int i=0; i<Writers; i++) delete writers[i];

			for (int i=0; i<Readers; i++) {
				latencies += readers[i]->sumLatency;
				nr += readers[i]->nread;
				// fmt::print(" - Reader {:>3d} latency {:>5.1f}us n {:>7d}\n", i, (readers[i]->sumLatency * 1.) / readers[i]->nread, readers[i]->nread);
			}
			fmt::print(" - Avg Read Latency: {:>5.1f}us, {:>7d} writes {:>7d} reads\n", (latencies * 1e-4) / (nr * 1e-4), nw, nr);
			for (int i=0; i<Readers; i++) delete readers[i];

		}
	};

	void run_futex_bm() {
		for (int i=0; i<20; i++) {
			{
				fmt::print("\n-------------------------------------------------------\n");
				fmt::print(" - futex condvar test\n");
				Benchmark<FutexCvHeader> bm;
				PrintContextSwitchesGuard pcsg;
				bm.run();
			}
			{
				fmt::print("\n-------------------------------------------------------\n");
				fmt::print(" - pthread condvar test\n");
				Benchmark<PthreadCvHeader> bm;
				PrintContextSwitchesGuard pcsg;
				bm.run();
			}
		}
	}

}

int main() {

	run_futex_bm();

	return 0;
}

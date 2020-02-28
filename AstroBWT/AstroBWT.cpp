/*
 * AstroBWT fast (but sometimes incorrect) implementation
 * This is just a test and debug harness, Windows only for now
 * Add "--bench" to the command line to benchmark performance
 *
 * Copyright 2020      SChernykh   <https://github.com/SChernykh>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <iostream>
#include <random>
#include <vector>
#include <thread>

#include <Windows.h>
#include "sais.h"
#include "sha3.h"
#include "Salsa20.hpp"

constexpr int STAGE1_SIZE = 147253;
constexpr int ALLOCATION_SIZE = (STAGE1_SIZE + 1048576) + (128 - (STAGE1_SIZE & 63));
constexpr int SCRATCHPAD_SIZE = ((ALLOCATION_SIZE * 17 + (1 << 21) - 1) >> 21) << 21;

constexpr int COUNTING_SORT_BITS = 10;
constexpr int COUNTING_SORT_SIZE = 1 << COUNTING_SORT_BITS;

BOOL SetLockPagesPrivilege();

__declspec(noinline) void sort_indices(int N, const uint8_t* v, uint64_t* indices, uint64_t* tmp_indices)
{
	uint32_t counters[2][COUNTING_SORT_SIZE] = {};

	for (int i = 0; i < N; ++i)
	{
		const uint64_t k = _byteswap_uint64(*reinterpret_cast<const uint64_t*>(v + i));
		++counters[0][(k >> (64 - COUNTING_SORT_BITS * 2)) & (COUNTING_SORT_SIZE - 1)];
		++counters[1][k >> (64 - COUNTING_SORT_BITS)];
	}

	uint32_t prev[2] = { counters[0][0], counters[1][0] };
	counters[0][0] = prev[0] - 1;
	counters[1][0] = prev[1] - 1;
	for (int i = 1; i < COUNTING_SORT_SIZE; ++i)
	{
		const uint32_t cur[2] = { counters[0][i] + prev[0], counters[1][i] + prev[1] };
		counters[0][i] = cur[0] - 1;
		counters[1][i] = cur[1] - 1;
		prev[0] = cur[0];
		prev[1] = cur[1];
	}

	for (int i = N - 1; i >= 0; --i)
	{
		const uint64_t k = _byteswap_uint64(*reinterpret_cast<const uint64_t*>(v + i));
		tmp_indices[counters[0][(k >> (64 - COUNTING_SORT_BITS * 2)) & (COUNTING_SORT_SIZE - 1)]--] = (k & (static_cast<uint64_t>(-1) << 21)) | i;
	}

	for (int i = N - 1; i >= 0; --i)
	{
		const uint64_t data = tmp_indices[i];
		indices[counters[1][data >> (64 - COUNTING_SORT_BITS)]--] = data;
	}

	auto smaller = [v](uint64_t a, uint64_t b)
	{
		const uint64_t value_a = a >> 21;
		const uint64_t value_b = b >> 21;

		if (value_a < value_b)
			return true;

		if (value_a > value_b)
			return false;

		const uint64_t data_a = _byteswap_uint64(*reinterpret_cast<const uint64_t*>(v + (a % (1 << 21)) + 5));
		const uint64_t data_b = _byteswap_uint64(*reinterpret_cast<const uint64_t*>(v + (b % (1 << 21)) + 5));
		return (data_a < data_b);
	};

	uint64_t prev_t = indices[0];
	for (int i = 1; i < N; ++i)
	{
		uint64_t t = indices[i];
		if (smaller(t, prev_t))
		{
			int j = i - 1;
			do
			{
				indices[j + 1] = prev_t;
				--j;
				if (j < 0)
					break;
				prev_t = indices[j];
			} while (smaller(t, prev_t));
			indices[j + 1] = t;
			t = indices[i];
		}
		prev_t = t;
	}
}

constexpr int N = STAGE1_SIZE + 1048575;

LARGE_INTEGER f, t1, t2, t3;

void test_sort_indices(int thread_index)
{
	SetLastError(0);
	SetThreadAffinityMask(GetCurrentThread(), 1ULL << thread_index);

	uint8_t node_number;
	GetNumaProcessorNode(thread_index, &node_number);

	std::mt19937 r;

	void* buf = VirtualAllocExNuma(GetCurrentProcess(), nullptr, SCRATCHPAD_SIZE, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE, node_number);
	if (!buf)
		buf = VirtualAllocExNuma(GetCurrentProcess(), nullptr, SCRATCHPAD_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node_number);

	uint8_t* allocation_ptr = (uint8_t*)buf;

	constexpr int N1 = ((N & ~63) + 64);

	uint8_t* v = allocation_ptr;
	allocation_ptr += N1;

	uint64_t* indices = (uint64_t*)allocation_ptr;
	allocation_ptr += N1 * sizeof(uint64_t);

	uint64_t* tmp_indices = (uint64_t*)allocation_ptr;
	int* sais_indices = (int*)tmp_indices;

	double dt = 0.0;
	double dt_sais = 0.0;
	int total_tests = 0;
	int wrong_tests = 0;

	for (int iter = 1;; ++iter)
	{
		for (int i = 0; i < N; ++i)
			v[i] = static_cast<uint8_t>(r());

		QueryPerformanceCounter(&t1);
		sort_indices(N, v, indices, tmp_indices);
		QueryPerformanceCounter(&t2);
		sais(v, sais_indices, N);
		QueryPerformanceCounter(&t3);

		for (int i = 0; i < N; ++i)
		{
			if ((indices[i] % (1 << 21)) != sais_indices[i])
			{
				++wrong_tests;
				break;
			}
		}

		++total_tests;
		dt += static_cast<double>(t2.QuadPart - t1.QuadPart) / f.QuadPart;
		dt_sais += static_cast<double>(t3.QuadPart - t2.QuadPart) / f.QuadPart;

		printf("Iteration %6d: sort_indices = %7.03f ms, SA-IS = %7.03f ms, %d incorrect\r", iter, (dt / total_tests) * 1000.0, (dt_sais / total_tests) * 1000.0, wrong_tests);
	}

	VirtualFree(buf, 0, MEM_RELEASE);
}

int decode_hex(char c)
{
	if ('0' <= c && c <= '9')
		return c - '0';
	if ('a' <= c && c <= 'f')
		return (c - 'a') + 10;
	return (c - 'A') + 10;
}

bool sha3_test()
{
	const char* tests[][2] = {
		{ "", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a" },
		{ "e9", "f0d04dd1e6cfc29a4460d521796852f25d9ef8d28b44ee91ff5b759d72c1e6d6" },
		{ "d477", "94279e8f5ccdf6e17f292b59698ab4e614dfe696a46c46da78305fc6a3146ab7" },
		{ "b053fa", "9d0ff086cd0ec06a682c51c094dc73abdc492004292344bd41b82a60498ccfdb" },
	};

	for (int i = 0; i < 4; ++i)
	{
		uint8_t input[4];
		uint8_t check[32];

		for (int j = 0; j < i; ++j)
			input[j] = (decode_hex(tests[i][0][j * 2]) << 4) | decode_hex(tests[i][0][j * 2 + 1]);
		for (int j = 0; j < 32; ++j)
			check[j] = (decode_hex(tests[i][1][j * 2]) << 4) | decode_hex(tests[i][1][j * 2 + 1]);

		uint8_t buf[32];
		sha3_HashBuffer(256, SHA3_FLAGS_NONE, input, i, buf, sizeof(buf));

		if (memcmp(buf, check, sizeof(buf)) != 0)
		{
			std::cout << "SHA3 test " << (i + 1) << " failed" << std::endl;
			return false;
		}
	}

	return true;
}

void bwt(const char* input, char* output)
{
	int len = static_cast<int>(strlen(input));
	uint8_t* v = new uint8_t[len + 16];
	memcpy(v, input, len);
	*(__m128i*)(v + len) = _mm_setzero_si128();

	uint64_t* indices = new uint64_t[len + 1];
	uint64_t* tmp_indices = new uint64_t[len + 1];

	sort_indices(len + 1, v, indices, tmp_indices);

	for (int i = 0; i < len + 1; ++i)
	{
		int index = indices[i] & ((1 << 21) - 1);
		if (index > 0)
			output[i] = v[index - 1];
		else
			output[i] = '$';
	}
	output[len + 1] = '\0';
}

bool bwt_test()
{
	const char* tests[][2] = {
		{ "banana", "annb$aa" },
		{ "abracadabra", "ard$rcaaaabb" },
		{ "appellee", "e$elplepa" },
		{ "GATGCGAGAGATG", "GGGGGGTCAA$TAA" },
	};

	for (int i = 0; i < 4; ++i)
	{
		char buf[32];
		bwt(tests[i][0], buf);
		if (strcmp(buf, tests[i][1]))
		{
			std::cout << "BWT test " << (i + 1) << " failed" << std::endl;
			return false;
		}
	}

	return true;
}

bool astrobwt_test()
{
	uint8_t key[32];
	uint8_t check_key[32];
	uint8_t check_str[] = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a";
	for (int i = 0; i < 32; ++i)
		check_key[i] = (decode_hex(check_str[i * 2]) << 4) | decode_hex(check_str[i * 2 + 1]);

	sha3_HashBuffer(256, SHA3_FLAGS_NONE, nullptr, 0, key, sizeof(key));

	if (memcmp(key, check_key, sizeof(key)) != 0)
	{
		std::cout << "AstroBWT test failed" << std::endl;
		return false;
	}

	uint8_t check_stage1[64];
	uint8_t check_stage1_str[] = "a683f2bb64611420c209db3bacb17637a0e841fa7791fd5721e7692e610a525d80b83d1ae4e61158f0c2f6ccf594c5815c0384a915bd5fbe8d309a4a34bbd611";
	for (int i = 0; i < 64; ++i)
		check_stage1[i] = (decode_hex(check_stage1_str[i * 2]) << 4) | decode_hex(check_stage1_str[i * 2 + 1]);

	const uint64_t iv = 0;

	std::vector<uint8_t> buf(SCRATCHPAD_SIZE);
	uint8_t* p = buf.data();

	uint8_t* stage1_output = p;
	p += STAGE1_SIZE + 64;
	{
		ZeroTier::Salsa20 s(key, &iv);
		s.XORKeyStream(stage1_output, STAGE1_SIZE);
		*(__m128i*)(stage1_output + STAGE1_SIZE) = _mm_setzero_si128();
	}

	if (memcmp(stage1_output + STAGE1_SIZE - sizeof(check_stage1), check_stage1, sizeof(check_stage1)) != 0)
	{
		std::cout << "AstroBWT test failed" << std::endl;
		return false;
	}

	uint64_t* indices = (uint64_t*) p;
	p += (STAGE1_SIZE + 1048575 + 1 + 64) * sizeof(uint64_t);
	uint64_t* tmp_indices = (uint64_t*) p;
	p += (STAGE1_SIZE + 1048575 + 1 + 64) * sizeof(uint64_t);

	sort_indices(STAGE1_SIZE + 1, stage1_output, indices, tmp_indices);

	uint8_t* stage1_result = (uint8_t*)(tmp_indices);
	for (int i = 0; i <= STAGE1_SIZE; ++i)
	{
		int index = indices[i] & ((1 << 21) - 1);
		if (index > 0)
			stage1_result[i] = stage1_output[index - 1];
		else
			stage1_result[i] = '\0';
	}

	sha3_HashBuffer(256, SHA3_FLAGS_NONE, stage1_result, STAGE1_SIZE + 1, key, sizeof(key));

	uint8_t check_key2[32];
	uint8_t check_str2[] = "99a0cdad2ecf83caa3f07328965adac86d4eab23ef8e0d4757fa935ff52cf063";
	for (int i = 0; i < 32; ++i)
		check_key2[i] = (decode_hex(check_str2[i * 2]) << 4) | decode_hex(check_str2[i * 2 + 1]);

	if (memcmp(key, check_key2, sizeof(key)) != 0)
	{
		std::cout << "AstroBWT test failed" << std::endl;
		return false;
	}

	int stage2_size = STAGE1_SIZE + (*(uint32_t*)(key) & 0xfffff);
	if (stage2_size != 1040334)
	{
		std::cout << "AstroBWT test failed" << std::endl;
		return false;
	}

	uint8_t check_stage2[64];
	uint8_t check_stage2_str[] = "09866f039730b5e3929df4b3fb1f284a76f9a33916660d9ddf2b5f6b9908b89f63871c69d3c6d6e928d21edcf2fa7bb85e6bbddb2626b94f904ed61740d46d94";
	for (int i = 0; i < 64; ++i)
		check_stage2[i] = (decode_hex(check_stage2_str[i * 2]) << 4) | decode_hex(check_stage2_str[i * 2 + 1]);

	uint8_t* stage2_output = p;
	{
		ZeroTier::Salsa20 s(key, &iv);
		s.XORKeyStream(stage2_output, stage2_size);
		*(__m128i*)(stage2_output + stage2_size) = _mm_setzero_si128();
	}

	if (memcmp(stage2_output + stage2_size - sizeof(check_stage2), check_stage2, sizeof(check_stage2)) != 0)
	{
		std::cout << "AstroBWT test failed" << std::endl;
		return false;
	}

	sort_indices(stage2_size + 1, stage2_output, indices, tmp_indices);

	uint8_t* stage2_result = (uint8_t*)(tmp_indices);
	for (int i = 0; i <= stage2_size; ++i)
	{
		int index = indices[i] & ((1 << 21) - 1);
		if (index > 0)
			stage2_result[i] = stage2_output[index - 1];
		else
			stage2_result[i] = '\0';
	}

	sha3_HashBuffer(256, SHA3_FLAGS_NONE, stage2_result, stage2_size + 1, key, sizeof(key));

	uint8_t check_key3[32];
	uint8_t check_str3[] = "cfc5155f4119ff5e11c71e907ce6708e1d39f0fe08b3a88ecb949abcc0d80e50";
	for (int i = 0; i < 32; ++i)
		check_key3[i] = (decode_hex(check_str3[i * 2]) << 4) | decode_hex(check_str3[i * 2 + 1]);

	if (memcmp(key, check_key3, sizeof(key)) != 0)
	{
		std::cout << "AstroBWT test failed" << std::endl;
		return false;
	}

	return true;
}

void astrobwt(const void* input_data, uint32_t input_size, void* scratchpad, uint8_t (&output_hash)[32])
{
	uint8_t key[32];
	uint8_t* scratchpad_ptr = (uint8_t*)(scratchpad) + 64;
	uint8_t* stage1_output = scratchpad_ptr;
	uint8_t* stage2_output = scratchpad_ptr;
	uint64_t* indices = (uint64_t*)(scratchpad_ptr + ALLOCATION_SIZE);
	uint64_t* tmp_indices = (uint64_t*)(scratchpad_ptr + ALLOCATION_SIZE * 9);
	uint8_t* stage1_result = (uint8_t*)(tmp_indices);
	uint8_t* stage2_result = (uint8_t*)(tmp_indices);

	sha3_HashBuffer(256, SHA3_FLAGS_NONE, input_data, input_size, key, sizeof(key));

	{
		const uint64_t iv = 0;
		ZeroTier::Salsa20 s(key, &iv);
		s.XORKeyStream(stage1_output, STAGE1_SIZE);
		*(__m128i*)(stage1_output + STAGE1_SIZE) = _mm_setzero_si128();
	}

	sort_indices(STAGE1_SIZE + 1, stage1_output, indices, tmp_indices);

	{
		const uint8_t* tmp = stage1_output - 1;
		for (int i = 0; i <= STAGE1_SIZE; ++i)
			stage1_result[i] = tmp[indices[i] & ((1 << 21) - 1)];
	}

	sha3_HashBuffer(256, SHA3_FLAGS_NONE, stage1_result, STAGE1_SIZE + 1, key, sizeof(key));

	const int stage2_size = STAGE1_SIZE + (*(uint32_t*)(key) & 0xfffff);
	{
		const uint64_t iv = 0;
		ZeroTier::Salsa20 s(key, &iv);
		s.XORKeyStream(stage2_output, stage2_size);
		*(__m128i*)(stage2_output + stage2_size) = _mm_setzero_si128();
	}

	sort_indices(stage2_size + 1, stage2_output, indices, tmp_indices);

	{
		const uint8_t* tmp = stage2_output - 1;
		int i = 0;
		const int n = ((stage2_size + 1) / 4) * 4;
		for (; i < n; i += 4)
		{
			stage2_result[i + 0] = tmp[indices[i + 0] & ((1 << 21) - 1)];
			stage2_result[i + 1] = tmp[indices[i + 1] & ((1 << 21) - 1)];
			stage2_result[i + 2] = tmp[indices[i + 2] & ((1 << 21) - 1)];
			stage2_result[i + 3] = tmp[indices[i + 3] & ((1 << 21) - 1)];
		}
		for (; i <= stage2_size; ++i)
			stage2_result[i] = tmp[indices[i] & ((1 << 21) - 1)];
	}

	sha3_HashBuffer(256, SHA3_FLAGS_NONE, stage2_result, stage2_size + 1, output_hash, sizeof(output_hash));
}

void astrobwt_bench(int thread_index, int total_threads, volatile long* thread_counter, double* hashrate)
{
	SetLastError(0);
	SetThreadAffinityMask(GetCurrentThread(), 1ULL << thread_index);

	uint8_t node_number;
	GetNumaProcessorNode(thread_index, &node_number);

	std::mt19937 r;

	void* scratchpad = VirtualAllocExNuma(GetCurrentProcess(), nullptr, SCRATCHPAD_SIZE, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE, node_number);
	if (!scratchpad)
		scratchpad = VirtualAllocExNuma(GetCurrentProcess(), nullptr, SCRATCHPAD_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node_number);

	uint8_t input[76];
	uint8_t hash[32];

	LARGE_INTEGER f, t1, t2;
	QueryPerformanceFrequency(&f);

	_InterlockedDecrement(thread_counter);
	do {} while (*thread_counter > 0);

	QueryPerformanceCounter(&t1);
	for (int iter = 0; iter < 100; ++iter)
	{
		for (int i = 0; i < sizeof(input); ++i)
			input[i] = static_cast<uint8_t>(r());

		astrobwt(input, sizeof(input), scratchpad, hash);
	}
	QueryPerformanceCounter(&t2);

	*hashrate = f.QuadPart * 100.0 / (t2.QuadPart - t1.QuadPart);

	_InterlockedIncrement(thread_counter);
	while (*thread_counter < total_threads)
	{
		for (int i = 0; i < sizeof(input); ++i)
			input[i] = static_cast<uint8_t>(r());

		astrobwt(input, sizeof(input), scratchpad, hash);
	}

	VirtualFree(scratchpad, 0, MEM_RELEASE);
}

int main(int argc, char** argv)
{
	QueryPerformanceFrequency(&f);
	SetLockPagesPrivilege();

	if (!sha3_test())
		return 1;

	if (!bwt_test())
		return 1;

	if (!astrobwt_test())
		return 1;

	{
		void* scratchpad = VirtualAlloc(nullptr, SCRATCHPAD_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
		uint8_t hash[32];
		astrobwt(nullptr, 0, scratchpad, hash);
		VirtualFree(scratchpad, 0, MEM_RELEASE);
		if (memcmp(hash, "\xcf\xc5\x15\x5f\x41\x19\xff\x5e\x11\xc7\x1e\x90\x7c\xe6\x70\x8e\x1d\x39\xf0\xfe\x08\xb3\xa8\x8e\xcb\x94\x9a\xbc\xc0\xd8\x0e\x50", sizeof(hash)) != 0)
		{
			std::cout << "astrobwt test failed" << std::endl;
			return 1;
		}
	}

	if ((argc > 1) && (strcmp(argv[1], "--bench") == 0))
	{
		std::vector<std::thread> threads;
		threads.reserve(std::thread::hardware_concurrency());

		for (size_t num_threads = 1; num_threads <= std::thread::hardware_concurrency(); ++num_threads)
		{
			threads.clear();
			std::vector<double> hashrate(num_threads);

			volatile long thread_counter = static_cast<long>(num_threads);
			for (int i = 0; i < num_threads; ++i)
				threads.emplace_back(astrobwt_bench, i, static_cast<int>(num_threads), &thread_counter, hashrate.data() + i);
			for (int i = 0; i < num_threads; ++i)
				threads[i].join();

			double total_hashrate = 0.0;
			for (int i = 0; i < num_threads; ++i)
				total_hashrate += hashrate[i];

			std::cout << num_threads << " threads: " << total_hashrate << " h/s" << std::endl;
		}
	}

	std::cout << "Testing sort_indices with length " << N << ": " << std::endl;
	test_sort_indices(0);

	return 0;
}

static BOOL SetLockPagesPrivilege() {
	HANDLE token;

	if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token) != TRUE) {
		return FALSE;
	}

	TOKEN_PRIVILEGES tp;
	tp.PrivilegeCount = 1;
	tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

	if (LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &(tp.Privileges[0].Luid)) != TRUE) {
		return FALSE;
	}

	BOOL rc = AdjustTokenPrivileges(token, FALSE, (PTOKEN_PRIVILEGES)&tp, 0, nullptr, nullptr);
	if (rc != TRUE || GetLastError() != ERROR_SUCCESS) {
		return FALSE;
	}

	CloseHandle(token);

	return TRUE;
}

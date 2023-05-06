#include <torch/extension.h>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>

namespace {

	using namespace torch;
	using namespace torch::indexing;

	torch::Tensor flatten_coords(torch::Tensor i, uint64_t lvl) {
		int N = i.size(1);
		int D = i.size(0);
		assert(D == 3);

		uint64_t S = (1lu<<lvl) - 1u;
		uint64_t X = S << (2*lvl);
		uint64_t Y = S << (1*lvl);
		uint64_t Z = S << (0*lvl);

		torch::Tensor out = torch::zeros_like(i.index({0}));

		auto it = thrust::make_counting_iterator(0);
		uint64_t* ii = (uint64_t*)i.data_ptr<int64_t>();
		thrust::transform(thrust::device,
				it, it + N,
				(ulong*)       out.data_ptr<int64_t>(),
				[ii,S,lvl,N]__device__(int i) {
					uint64_t x = ii[0*N+i];
					uint64_t y = ii[1*N+i];
					uint64_t z = ii[2*N+i];
					return  ((x & S) << (lvl*2)) |
							((y & S) << (lvl*1)) |
							((z & S) << (lvl*0)) ;
				});
		return out;
	}

	torch::Tensor unflatten_coords(torch::Tensor i, uint64_t lvl) {
		int N = i.size(0);

		uint64_t S = (1lu<<lvl) - 1u;

		torch::Tensor out = torch::zeros({3,N}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

		uint64_t* ii = (uint64_t*)  i.data_ptr<int64_t>();
		uint64_t* oi = (uint64_t*)out.data_ptr<int64_t>();
		auto it = thrust::make_counting_iterator(0);
		thrust::for_each(thrust::device,
				it,it+N,
				[ii,oi,N,S,lvl]__device__(int i) {
					uint64_t e = ii[i];
					oi[0*N+i] = (e >> lvl*2) & S;
					oi[1*N+i] = (e >> lvl*1) & S;
					oi[2*N+i] = (e >> lvl*0) & S;
				});

		return out;
	}


	// NOTE: In this version I've not applied the shifted weight to the coarse grid,
	//       so it is sort of 'blocky'. This version could have been done in python I think!
	// FIXME: Either use a weighting coming from the shift, or evaluate the F kernel for each point
	//        to get the non-blocky tent value.

	// ~~~I'll use the form: A_ij = <Ac_i, nabla{Af}_j~~~
	// Actually: A_ij = <grad[Af_i], grad[Ac_j]>
	torch::Tensor make_level_transfer(
			int levelF, int levelC,
			torch::Tensor indsF,
			torch::Tensor indsC,
			torch::Tensor gradStencilSt
			) {

		// Kernel width (not number of elements)
		int stencilSize = gradStencilSt.size(0);
		auto gradStencilInds = gradStencilSt.indices().contiguous() - stencilSize/2;
		// auto gradStencilVals = gradStencilSt.values().cpu();
		auto gradStencilVals = gradStencilSt.values();
		assert(gradStencilInds.size(0) == 3);
		assert(gradStencilVals.size(1) == 3);
		assert(gradStencilVals.size(0) == gradStencilInds.size(1));

		int nStencil = gradStencilInds.size(1);
		// auto gradStencilInds_ = gradStencilInds.accessor<long ,2>();
		// auto gradStencilValsCpu_ = gradStencilValsCpu.accessor<float,1>();

		assert((indsC < (1<<levelC)).all().item().to<bool>());
		assert((indsF < (1<<levelF)).all().item().to<bool>());

		assert(levelC <= levelF);
		// the divisor to go from F -> C
		int64_t levelDivisor = 1 << (levelF - levelC);

		std::cout << " - input sizes C=" << indsC.sizes() << " F=" << indsF.sizes() << " level divisor " << levelDivisor << std::endl;

		// NOTE: indsFC may not be sorted, but that is okay!
		// It is not sorted because if {x=5,y=2}, and the divisor is 2, then it becomes 2 instead of "2.5" and any point with {x=4, y>2} will have moved in front!
		// It is okay because we never change the order, so indsFF and indsFC correspond element by element!

		// [3,N] -> [N]
		std::cout << " - flattening coords." << std::endl;
		// torch::Tensor indsCC = flatten_coords(indsC, levelC);
		torch::Tensor indsFF0 = flatten_coords(indsF, levelF);
		// torch::Tensor indsFC = flatten_coords(indsF.div(levelDivisor, "floor"), levelC);
		int NC = indsC.size(1);
		int NF = indsF.size(1);
		// assert(NF == indsFC.size(0));


		std::cout << " - verifying sorted (after flatten)." << std::endl;
		// std::cout << " - indsF:\n" << indsF;
		// std::cout << " - indsF/divisor:\n" << indsF.div(levelDivisor, "floor");
		// std::cout << " - indsFF:\n" << indsFF;
		// std::cout << " - indsFC:\n" << indsFC;
		// assert(thrust::is_sorted(thrust::device, (uint64_t*)indsCC.data_ptr<int64_t>(), (uint64_t*)indsCC.data_ptr<int64_t>()+NC));
		// assert(thrust::is_sorted(thrust::device, (uint64_t*)indsFF.data_ptr<int64_t>(), (uint64_t*)indsFF.data_ptr<int64_t>()+NF));
		// assert(thrust::is_sorted(thrust::device, (uint64_t*)indsFC.data_ptr<int64_t>(), (uint64_t*)indsFC.data_ptr<int64_t>()+NF));
		// std::cout << " - indsCC:\n" << indsCC;
		// std::cout << " - indsFC:\n" << indsFC;


		torch::Tensor outInds = torch::zeros({2,0}, torch::TensorOptions().dtype(torch::kLong) .device(torch::kCUDA));
		torch::Tensor outVals = torch::zeros({0}  , torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

		// Batch compute the inner product of the gradients.
		auto gi_gj = (gradStencilVals.view({-1,1,3}) * gradStencilVals.view({1,-1,3})).sum(2);
		auto gi_gj_cpu = gi_gj.cpu();
		auto gi_gj_cpu_acc = gi_gj_cpu.accessor<float,2>();

		int nActualLoops = 0;
		std::cout << " - running " << nStencil*nStencil << " loops." << std::endl;
		for (int i=0; i<nStencil; i++) {
			torch::Tensor di = gradStencilInds.index({Slice(),Slice(i,i+1)});
			torch::Tensor gi = gradStencilVals.index({i});
			for (int j=0; j<nStencil; j++) {
				torch::Tensor dj = gradStencilInds.index({Slice(),Slice(j,j+1)});
				torch::Tensor gj = gradStencilVals.index({j});

				// NOTE: severe aliasing. Should have vectorial values that are interpolated with fine/coarse grid.
				// float v = (gi * gj).sum().item().to<float>();
				float v = gi_gj_cpu_acc[i][j];
				if (v > -1e-9 and v < 1e-9) {
					// std::cout << " - skipping iter " << i << " " << j << "\n";
					continue;
				} else {
					nActualLoops++;
				}


				/*
				torch::Tensor hits = torch::zeros({NF}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

				// Find the next coord in the C grid from the F grid.
				thrust::lower_bound(thrust::device,
						(const uint64_t*)indsFC.data_ptr<int64_t>(),
						(const uint64_t*)indsFC.data_ptr<int64_t>() + NC,
						(const uint64_t*)indsCC.data_ptr<int64_t>(),
						(const uint64_t*)indsCC.data_ptr<int64_t>() + NF,
						hits.data_ptr<int64_t>(),
						thrust::less<uint64_t>());

				// Now we can use them to check if we have a hit in the fine grid.
				// i.e. if the two grids had the same scale, we'd want equality.
				// But with differing grids, multiple fine cells may map to one coarse cell.

				// We'll subselect this.
				torch::Tensor addInds = torch::stack({ indsFC, hits }, 0);

				// torch::Tensor addVals; // I think this should be stencil times something based on offset... FIXME:
				torch::Tensor addVals = torch::full({NF}, v, torch::TensorOptions().device(torch::kCUDA));

				*/

				auto indsFF_ = indsF.add(di);
				auto indsCC_ = indsC.add(dj);
				// auto indsFF = flatten_coords(indsFF_, levelF);
				auto indsFC_div = indsFF_.div_(levelDivisor, "floor");
				auto indsFC_shifted = flatten_coords(indsFC_div, levelC);
				// auto indsCC_shifted = indsCC.sub(dj);
				auto indsCC_shifted = flatten_coords(indsCC_, levelC);

				// i^ = i + di
				// j^ = j + dj, then
				// j = i + di - dj
				// These should be be in the original indsC set (or otherwise masked out later on)
				auto indsFC_unshifted = flatten_coords(indsFC_div.sub(dj), levelC);



				// Actually the binary_search fn looks like a good fit here.
				//
				// convert f -> c, then mask out invalid hits, so that the final masked array contains valid (f,c) hits.
				torch::Tensor mask = torch::zeros({NF}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
				thrust::binary_search(thrust::device,
						(const uint64_t*)indsFC_shifted.data_ptr<int64_t>(),
						(const uint64_t*)indsFC_shifted.data_ptr<int64_t>() + NF,
						(const uint64_t*)indsCC_shifted.data_ptr<int64_t>(),
						(const uint64_t*)indsCC_shifted.data_ptr<int64_t>() + NC,
						mask.data_ptr<bool>(),
						thrust::less<uint64_t>());

				// TODO: Test me.
				// std::cout << " - indsFF_:\n" << indsFF_ << "\n";
				// I don't think we need to mask off indsFC_ because we only have an output when we match with FF_, which is
				// verified here. (We can't easily mask the mask variable with a check on indsFC_ because it is another size)
				mask = mask.logical_and((indsFF_ >= 0) & (indsFF_ < (1<<levelC))).all(0);
				// mask = mask & ((indsFC_ >= 0) & (indsFC_ < (1<<levelC))).all(0);
				// mask = mask & ((indsCC_ >= 0) & (indsCC_ < (1<<levelC))).all(0);

				// We'll subselect this.
				// torch::Tensor addInds = torch::stack({ indsFF, indsFC_shifted }, 0);
				torch::Tensor addInds = torch::stack({ indsFF0, indsFC_unshifted }, 0);

				// torch::Tensor addVals; // I think this should be stencil times something based on offset... FIXME:
				// torch::Tensor addVals = torch::full({NF}, v, torch::TensorOptions().device(torch::kCUDA));
				torch::Tensor addVals = gi_gj.index({Slice(i,i+1),j}).repeat({NF});

				addInds = addInds.masked_select(mask.view({1,-1})).view({2,-1});
				addVals = addVals.masked_select(mask);
				if (addInds.numel() > 0) {
					// std::cout << " cat i " << outInds.sizes() << addInds.sizes() << "\n";
					// std::cout << " cat v " << outVals.sizes() << addVals.sizes() << "\n";
					// std::cout << " - now have num entries " << outVals.size(0) << "\n";
					// std::cout << " - assosciate:\n" << torch::stack({unflatten_coords(addInds.index({0}),levelF), unflatten_coords(addInds.index({1}),levelC)},1) << "\n";
				}
				outInds = torch::cat({outInds,addInds},1);
				outVals = torch::cat({outVals,addVals},0);
			}
		}
		std::cout << " - num loops " << nActualLoops << " / " << (nStencil*nStencil) << "\n";

		// TODO: We have the FLATTENED-INDICES as the elements.
		// For the coming sparse matrix multiplies, we need the VECTOR-INDICES of the SPATIAL-INDICES.
		// We can do this with thrust by using lower_bound on the mapped indices.
		int outN = outInds.size(1);
		assert(thrust::is_sorted(thrust::device, (uint64_t*)indsFF0.data_ptr<int64_t>(), (uint64_t*)indsFF0.data_ptr<int64_t>()+NF));

		// torch::Tensor outInds2 = torch::zeros_like(outInds);

		// the FF indices.
		thrust::lower_bound(thrust::device,
				(uint64_t*)indsFF0.data_ptr<int64_t>(), (uint64_t*)indsFF0.data_ptr<int64_t>()+indsFF0.numel(),
				(uint64_t*)outInds.data_ptr<int64_t>(), (uint64_t*)outInds.data_ptr<int64_t>()+outN,
				(uint64_t*)outInds.data_ptr<int64_t>(),
				thrust::less<uint64_t>());

		// the FC indices.
		torch::Tensor indsCC0 = flatten_coords(indsC, levelC);
		assert(thrust::is_sorted(thrust::device, (uint64_t*)indsCC0.data_ptr<int64_t>(), (uint64_t*)indsCC0.data_ptr<int64_t>()+indsCC0.numel()));
		thrust::lower_bound(thrust::device,
				(uint64_t*)indsCC0.data_ptr<int64_t>(), (uint64_t*)indsCC0.data_ptr<int64_t>()+indsCC0.numel(),
				(uint64_t*)outInds.data_ptr<int64_t>()+outN, (uint64_t*)outInds.data_ptr<int64_t>()+outN*2,
				(uint64_t*)outInds.data_ptr<int64_t>()+outN,
				thrust::less<uint64_t>());


		return torch::sparse_coo_tensor(outInds,outVals).coalesce();
		// return torch::sparse_coo_tensor(outInds,outVals);

	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_level_transfer", &make_level_transfer, "make_level_transfer");
}

#include <torch/extension.h>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

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


		/*
		auto ii = i.t().contiguous();
		thrust::transform(thrust::device,
				(const ulong3*) ii.data_ptr<int64_t>(),
				(const ulong3*) (ii.data_ptr<int64_t>()+N*D),
				(ulong*)       out.data_ptr<int64_t>(),
				[S,lvl]__device__(ulong3 xyz) {
					return  ((xyz.x & S) << (lvl*2)) |
							((xyz.y & S) << (lvl*1)) |
							((xyz.z & S) << (lvl*0)) ;
				});
		*/
		auto it = thrust::make_counting_iterator(0);
		uint64_t* ii = (uint64_t*)i.data_ptr<int64_t>();
		thrust::transform(thrust::device,
				it, it + N*D,
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



	// I'll use the form: A_ij = <Ac_i, nabla{Af}_j
	torch::Tensor make_level_transfer(
			int levelC, int levelF,
			torch::Tensor indsC,
			torch::Tensor indsF,
			torch::Tensor lapStencilStCpu
			) {

		// auto cooC_inds = cooC.indices();
		// auto cooC_vals = cooC.values();
		// auto cooF_inds = cooF.indices();
		// auto cooF_vals = cooF.values();

		int stencilSize = lapStencilStCpu.size(0);
		auto lapStencilInds = lapStencilStCpu.indices().t().contiguous() - stencilSize/2;
		auto lapStencilValsCpu = lapStencilStCpu.values().cpu();

		int nStencil = lapStencilInds.size(0);
		// auto lapStencilInds_ = lapStencilInds.accessor<long ,2>();
		auto lapStencilValsCpu_ = lapStencilValsCpu.accessor<float,1>();

		assert((indsC < (1<<levelC)).all().item().to<bool>());
		assert((indsF < (1<<levelF)).all().item().to<bool>());

		assert(levelC < levelF);
		// the divisor to go from F -> C
		int64_t levelDivisor = 1 << (levelF - levelC);

		std::cout << " - input sizes C=" << indsC.sizes() << " F=" << indsF.sizes() << " level divisor " << levelDivisor << std::endl;

		// NOTE: indsFC may not be sorted, but that is okay!
		// It is not sorted because if {x=5,y=2}, and the divisor is 2, then it becomes 2 instead of "2.5" and any point with {x=4, y>2} will have moved in front!
		// It is okay because we never change the order, so indsFF and indsFC correspond element by element!

		// [3,N] -> [N]
		std::cout << " - flattening coords." << std::endl;
		torch::Tensor indsCC = flatten_coords(indsC, levelC);
		torch::Tensor indsFF = flatten_coords(indsF, levelF);
		torch::Tensor indsFC = flatten_coords(indsF.div(levelDivisor, "floor"), levelC);
		int NC = indsCC.size(0);
		int NF = indsFF.size(0);
		assert(NF == indsFC.size(0));


		std::cout << " - verifying sorted (after flatten)." << std::endl;
		// std::cout << " - indsF:\n" << indsF;
		// std::cout << " - indsF/divisor:\n" << indsF.div(levelDivisor, "floor");
		// std::cout << " - indsFF:\n" << indsFF;
		// std::cout << " - indsFC:\n" << indsFC;
		assert(thrust::is_sorted(thrust::device, (uint64_t*)indsCC.data_ptr<int64_t>(), (uint64_t*)indsCC.data_ptr<int64_t>()+NC));
		assert(thrust::is_sorted(thrust::device, (uint64_t*)indsFF.data_ptr<int64_t>(), (uint64_t*)indsFF.data_ptr<int64_t>()+NF));
		// assert(thrust::is_sorted(thrust::device, (uint64_t*)indsFC.data_ptr<int64_t>(), (uint64_t*)indsFC.data_ptr<int64_t>()+NF));
		std::cout << " - indsCC:\n" << indsCC;
		std::cout << " - indsFC:\n" << indsFC;


		torch::Tensor outInds = torch::zeros({2,0}, torch::TensorOptions().dtype(torch::kLong) .device(torch::kCUDA));
		torch::Tensor outVals = torch::zeros({0}  , torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

		std::cout << " - running " << nStencil << " loops." << std::endl;
		for (int i=0; i<nStencil; i++) {
			// Make sure you call cpu() on the accessor.
			float v = lapStencilValsCpu_[i];
			// long dx = lapStencilInds_[i][0] - stencilSize/2;
			// long dy = lapStencilInds_[i][1] - stencilSize/2;
			// long dz = lapStencilInds_[i][2] - stencilSize/2;

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

			// FIXME: Oh, I need to shift on the fine grid BEFORE query...

			// Actually the binary_search fn looks like a good fit here.
			torch::Tensor mask = torch::zeros({NF}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
			thrust::binary_search(thrust::device,
					(const uint64_t*)indsFC.data_ptr<int64_t>(),
					(const uint64_t*)indsFC.data_ptr<int64_t>() + NF,
					(const uint64_t*)indsCC.data_ptr<int64_t>(),
					(const uint64_t*)indsCC.data_ptr<int64_t>() + NC,
					mask.data_ptr<bool>(),
					thrust::less<uint64_t>());

			// We'll subselect this.
			torch::Tensor addInds = torch::stack({ indsFF, indsFC }, 0);

			// torch::Tensor addVals; // I think this should be stencil times something based on offset... FIXME:
			torch::Tensor addVals = torch::full({NF}, v, torch::TensorOptions().device(torch::kCUDA));

			addInds = addInds.masked_select(mask.view({1,-1})).view({2,-1});
			addVals = addVals.masked_select(mask);
			std::cout << " cat i " << outInds.sizes() << addInds.sizes() << "\n";
			std::cout << " cat v " << outVals.sizes() << addVals.sizes() << "\n";
			outInds = torch::cat({outInds,addInds},1);
			outVals = torch::cat({outVals,addVals},0);
			std::cout << " - now have num entries " << outVals.size(0) << "\n";

		}

		// TODO: We have the FLATTENED-INDICES as the elements.
		// For the coming sparse matrix multiplies, we need the VECTOR-INDICES of the SPATIAL-INDICES.

		return torch::sparse_coo_tensor(outInds,outVals).coalesce();

	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_level_transfer", &make_level_transfer, "make_level_transfer");
}

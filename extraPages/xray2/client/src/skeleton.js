

// function make_instanced_skeleton(d) {
	// return regl({
	// });
// }

// NOTE: no dependnece on #joints l
function make_colors(t,n,l) {
	let c = [];
	for (let i=0; i<t*n; i++) {
	// for (let i=0; i<t*n*l; i++) {
		let r = Math.random();
		let g = Math.random();
		let b = Math.random();
		const a = .8;
		c.push(r / (r+g+b));
		c.push(g / (r+g+b));
		c.push(b / (r+g+b));
		c.push(a);
	}
	console.log(c);
	return c;
}

function hstack(a,b, M, N) {
	let c = new Float32Array(a.length + b.length);
	let j = 0;
	for (let i=0; i<a.length / M; i++) {
		for (let k=0; k<M; k++)
			c[j++] = a[i*M+k];
		for (let k=0; k<N; k++)
			c[j++] = b[i*N+k];
	}
	return c;
}

export function Skeleton(regl, d) {
	this.T = d.T || 1;
	this.N = d.N || 1;
	this.L = d.jointNames.length;
	this.L2 = 2 * d.jointNames.length;
	this.linewidth = d.linewidth || 1;
	this.timeIdx = 0;

	// this.indices = regl.buffer({data: new Uint16Array(d.indices), type: 'uint16'});
	this.jointNames = d.jointNames;

	const positions = d.positions;
	const colors = d.colors || make_colors(this.T, this.N, this.L);

	// this.buf = regl.buffer(hstack(positions, colors, 3, 4));
	this.pos_buf = regl.buffer(positions);
	this.col_buf = regl.buffer(colors);
	// this.off = regl.buffer([0,0,0, 1,0,0]);

	// NOTE: Unfortunately you cannot make this completely instanced because we want to keep the same indices
	// and advance the positions outside of the instancing.
	//
	// A workaround is to still use instancing, but duplicate the inds.
	// NOTE: NEvermind, that does not work.
	/*
	// this.indices = regl.elements(new Uint8Array(d.indices));
	let inds = d.indices.slice();
	for (let i=0; i<this.N; i++) {
		for (let j=0; j<this.L; j++) {
			inds.push(inds[j]);
		}
	}
	this.indices = regl.elements(new Uint8Array(inds));
	if (Skeleton.prototype.draw === undefined) {
		Skeleton.prototype.draw = regl({

				uniforms: {
					u_mvew: regl.context('mvew'),
					u_proj: regl.context('proj'),
				},

				vert: `precision mediump float;
				uniform mat4 u_proj;
				uniform mat4 u_mvew;
				attribute vec3 a_pos;
				attribute vec4 a_col;
				varying vec4 v_color;
				void main() {
					gl_Position = u_proj * u_mvew * vec4(a_pos, 1.);
					v_color = a_col;
				}`,
				frag: `precision mediump float;
				varying vec4 v_color;
				void main() {
					gl_FragColor = v_color;
				}`,

				attributes: {
					a_pos: { buffer: regl.this('pos_buf'), offset: 0, divisor: 0 },
					a_col: { buffer: regl.this('col_buf'), offset: 0, divisor: 1 },
					// a_col: { buffer: regl.this('col_buf'), offset:  0, divisor: regl.this('L') },
					// a_col: { buffer: function() { return this.col_buf; }, offset:  0, divisor: 1, size:4 },
					// a_col: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
				},

				// instances: regl.this('N'),
				instances: 2,
				elements: regl.this('indices'),
				// count: regl.this('L2'),
				primitive: 'lines'
		});
	}
	*/

	this.indices = regl.elements(new Uint8Array(d.indices));
	if (Skeleton.prototype.draw === undefined) {
		Skeleton.prototype.draw_ = regl({

				uniforms: {
					u_mvew: regl.context('mvew'),
					u_proj: regl.context('proj'),
				},

				vert: `precision mediump float;
				uniform mat4 u_proj;
				uniform mat4 u_mvew;
				attribute vec3 a_pos;
				attribute vec4 a_col;
				varying vec4 v_color;
				void main() {
					gl_Position = u_proj * u_mvew * vec4(a_pos, 1.);
					v_color = a_col;
				}`,
				frag: `precision mediump float;
				varying vec4 v_color;
				void main() {
					gl_FragColor = v_color;
				}`,

				attributes: {
					a_pos: { buffer: regl.this('pos_buf'), offset: (c,p) => 4*3*(this.L*this.N*p.timeIdx + this.L*p.skelIdx) },
					a_col: { buffer: regl.this('col_buf'), offset: (c,p) => 4*4*p.skelIdx, divisor: 1},
				},

				elements: regl.this('indices'),
				primitive: 'lines',
				lineWidth: regl.this('linewidth')
		});

		Skeleton.prototype.draw = function() {
			for (let i=0; i<this.N; i++) {
				this.draw_({timeIdx: this.timeIdx, skelIdx: i});
			}
		}
	}

}




export function create_dummy_skeleton(regl) {
	let jointNames = ["knee", "ankle", "lhip", "mhip"];
	let indices = [0,1, 0,2, 2,3];
	let positions = [
		-.3,0,0,
		-.3,-1,0,
		-.3,.8,.05,
		0,.9,.05,

		.3,0,.5,
		.3,-1,0,
		.3,.8,.05,
		0,.9,.05,

	];
	return new Skeleton(regl, {jointNames, indices, positions, T:1, N:2});
}

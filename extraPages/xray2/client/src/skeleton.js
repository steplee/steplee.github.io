import {mm,mv} from './math3d.js';

// function make_instanced_skeleton(d) {
	// return regl({
	// });
// }

//
// A `Skeleton` is created using the below described parameter object. The class prototype has a `draw` method.
// The draw method expects `mvew` and `proj` matrices in the context, and `timeIdx` as a prop.
//
// The constructor parameter object must be like:
//	 - jointNames: an array of strings, whose order should be consistent with other fields
//	 - T: the length of the time axis
//	 - N: the length of the batch axis
//	 - positions: the float32 array buffer of vertices, must be formated like [T, N, L, 3]
//	 - (optional) colors: colors per vertex
//

// ~NOTE: no dependnece on #joints l~
// NOTE: no dependnece on #T
function make_colors(t,n,l) {
	let c = [];
	for (let i=0; i<n; i++) {
	// for (let i=0; i<t*n*l; i++) {
		let r = Math.random();
		let g = Math.random();
		let b = Math.random();
		const a = .8;
		for (let j=0; j<l; j++) {
		c.push(r / (r+g+b));
		c.push(g / (r+g+b));
		c.push(b / (r+g+b));
		c.push(a);
		}
	}
	console.log('created colors', c);
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

function create_labels(k, jointNames) {
	let container = document.getElementById('overText' + k);
	console.log(container);

	if (container === undefined || container === null) {
		console.log(container);
		container = document.createElement('div');
		container.id = 'overText' + k;
		container.className = 'overTextContainer';
		document.getElementById('overTextTop').append(container);

		for (let name of jointNames) {
			let p = document.createElement('p');
			p.id = k + name;
			p.innerHTML = name;
			p.className = 'overText';
			container.append(p);
			console.log('add', k, name);
		}
	}

	return container;
}

function create_buttons(k) {

	const h = `
					<div class="controlName">${k}</div>
					<input type="checkbox" checked>
					<input type="checkbox">
				`.trim();

	let container = document.getElementById("buttons" + k);

	if (container === null) {
		let holder = document.getElementById('showButtons');
		container = document.createElement('div');

		container.className = 'controlRow'
		container.id = "buttons" + k;
		container.innerHTML = h;

		holder.appendChild(container);
	} else {
		// to remove event listeners
		let oldContainer = container;
		container = oldContainer.cloneNode(true);
		oldContainer.replaceWith(container);
	}
	return [container.children[1], container.children[2]]
}


export function Skeleton(regl, name, d) {
	this.T = d.T;
	this.N = d.N;
	this.L = d.jointNames.length;
	this.L2 = 2 * d.jointNames.length;
	this.linewidth = d.linewidth || 1;
	this.timeIdx = 0;


	// this.indices = regl.buffer({data: new Uint16Array(d.indices), type: 'uint16'});
	this.jointNames = d.jointNames;

	this.labelContainer = create_labels(name, this.jointNames);
	[this.buttonActive, this.buttonShowLabels] = create_buttons(name);

	this.active = true;
	this.showLabels = false;
	let this_ = this;
	this.buttonActive    .addEventListener('change', (a) => {console.log(this_.active); this_.active = a.target.checked});
	this.buttonShowLabels.addEventListener('change', (a) => {this_.showLabels = a.target.checked});
	

	const positions = d.positions;
	const colors = d.colors || make_colors(this.T, this.N, this.L);

	// this.buf = regl.buffer(hstack(positions, colors, 3, 4));
	this.positions = positions;
	this.pos_buf = regl.buffer(positions);
	this.col_buf = regl.buffer(colors);
	console.log(this)
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
					// a_pos: { buffer: regl.this('pos_buf'), offset: (c,p) => 4*3*(this.L*this.N*p.timeIdx + this.L*p.skelIdx) },
					a_pos: { buffer: regl.this('pos_buf'), offset: (c,p) => {
						// console.log(this, this.T, this.L, this.N, p.timeIdx, p.skelIdx, 'off', 4*3*(this.L*this.N*p.timeIdx + this.L*p.skelIdx) );
						// return 4*3*(this.L*this.N*p.timeIdx + this.L*p.skelIdx)
						return 4*3*(p.offset)
					}
					},
					// a_col: { buffer: regl.this('col_buf'), offset: (c,p) => 4*4*p.skelIdx, divisor: 1},
					a_col: { buffer: regl.this('col_buf'), offset: (c,p) => 4*4*(this.L*p.skelIdx) },
					// a_col: { buffer: regl.this('col_buf'), offset: (c,p) => 4*4*(p.offset) },
				},

				elements: regl.this('indices'),
				primitive: 'lines',
				lineWidth: regl.this('linewidth')
		});

		Skeleton.prototype.draw = function(ctx, p) {
			if (!this.active) return;

			for (let i=0; i<this.N; i++) {
				console.log(`draw t=${p.timeIdx}, n=${i}`)

				const offset = this.L*this.N*p.timeIdx + this.L*i;
				this.draw_({timeIdx: p.timeIdx, skelIdx: i, offset: offset});


				// if (p.showLabels) {
				if (this.showLabels) {
					// Use skelIdx = 0.

					let mvp = mm(ctx.proj, ctx.mvew);
					let o = p.timeIdx*this.L*this.N*3 + 3*this.L*i;
					this.labelContainer.style.display = "block";

					for (let l=0; l<this.L; l++) {
						let a = [ this.positions[o + l*3+0], this.positions[o + l*3+1], this.positions[o + l*3+2], 1];
						let b = mv(mvp,a);
						let c = [b[0]/b[3], b[1]/b[3], b[2]/b[3]];
						let x = ( c[0] * .5 + .5) * ctx.viewportWidth;
						let y = (-c[1] * .5 + .5) * ctx.viewportHeight;

						let label = this.labelContainer.children.item(l);
						label.style.top = y;
						label.style.left = x;
					}
				} else {
					this.labelContainer.style.display = "none";
				}
				// let p = this.pos_buf.
				// console.log(mvp);
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
	return new Skeleton(regl, 'dummy', {jointNames, indices, positions, T:1, N:2});
}

export function create_skeleton_from_json(regl, name, json) {
	return new Skeleton(regl, name, json);
}



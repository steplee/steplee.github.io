import regl_ from "regl";
import math from "./math3d.js";
import {create_dummy_skeleton, create_skeleton_from_json, Skeleton} from "./skeleton.js";

export function hi() {
	console.log('hi2')
}

window.hi = hi;

function create_axes(regl) {

	let buf = regl.buffer(new Float32Array([
		0,0,0, 1,0,0,.8,
		1,0,0, 1,0,0,.8,
		0,0,0, 0,1,0,.8,
		0,1,0, 0,1,0,.8,
		0,0,0, 0,0,1,.8,
		0,0,1, 0,0,1,.8,
	]));

	return regl({
		vert: `
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
			a_pos: { buffer: buf, offset:  0, stride: 4*(3+4) },
			a_col: { buffer: buf, offset: 12, stride: 4*(3+4) },
		},
		uniforms: {
			u_mvew: regl.context('mvew'),
			u_proj: regl.context('proj'),
		},
		primitive: 'lines',
		count: 6,
	});

}

/*
function create_skeleton(regl) {

	let inds_per_skel = 25;

	let buf = regl.buffer(new Float32Array([
		0,0,0, 1,0,0,.8,
		1,0,0, 1,0,0,.8,
		0,0,0, 0,1,0,.8,
		0,1,0, 0,1,0,.8,
		0,0,0, 0,0,1,.8,
		0,0,1, 0,0,1,.8,
	]));

	regl({
		vert: `
		uniform mat4 u_proj;
		uniform mat4 u_mvew;
		attribute vec3 a_pos;
		attribute vec4 a_col;
		varying vec4 v_color;
		void main() {
			gl_Position = u_proj * u_mvew * vec4(a_pos, 1.);
			v_color = a_col;
		}`,
		frag: `
		varying vec4 v_color;
		void main() {
			gl_FragColor = v_color;
		}`,
		elements: elements,
		attributes: {
			a_pos: { buffer: buf, offset:  0, stride: 4*(3+4) },
			a_col: { buffer: buf, offset: 12, stride: 4*(3+4) },
		},
		uniforms: {
			u_mvew: regl.context('mvew'),
			u_proj: regl.context('proj'),
		},
		primitive: 'lines',
		count: inds_per_skel,
		offset: (ctx,props) => props.idx * inds_per_skel,
	});
}
*/

function get_skeletons(form) {
	let formData = undefined;
	if (form) {
		let d = new FormData(form);
		// console.log(d)
		formData = JSON.stringify(Object.fromEntries(d));
		// formData = new FormData(form);
	}
	console.log('formData', formData);

	fetch('/skeleton', {method: 'post', body: formData})
		.then((response) => response.json())
		.then((d) => {
			console.log(d)
			let o = {};
			for (let skelName in d.skeletons) {
				o[skelName] = create_skeleton_from_json(window.app.regl, skelName, d.skeletons[skelName]);
			}
			console.log("setting new skeletons", o);
			window.app.setSkeletons(o)
		});

	// window.app.setSkeletons({
		// dummy: create_dummy_skeleton(window.app.regl)
	// });

}
function get_frame(i) {
}

function App(regl) {
		this.sleepTime = 16; // 60 fps default.

		window.app = this;
		this.regl = regl;
		this.setupScene = regl({
			context: {
				// cam: cam,
				proj: (ctx,props) => props.cam.proj,
				mvew: (ctx,props) => props.cam.view
			}
		});

		this.cam = new math.FreeCamera(container);
		this.axes = create_axes(regl);
		this.showLabels = false;

		this.skeletons = {};
		this.maxT = 1;
		this.maxN = 1;
		this.curT = 0;
		this.curN = 0;

		let sliderSelectedFrame = document.getElementById('selectedFrame');
		let sliderSelectedIdx = document.getElementById('selectedIdx');
		let checkboxShowLabels = document.getElementById('showLabels');
		let buttonRefresh = document.getElementById('refresh');
		let theForm = document.getElementById('theForm');
		checkboxShowLabels.checked = this.showLabels;
		checkboxShowLabels .addEventListener('change', (a) => {this.showLabels = a.target.checked});
		sliderSelectedFrame.addEventListener('input', (a) => {this.curT = a.target.value});
		sliderSelectedIdx  .addEventListener('input', (a) => {this.curN = a.target.value});
		buttonRefresh      .addEventListener('click', (a) => {get_skeletons(theForm)});


		let this_ = this;
		this.setSkeletons = function(d) {
			this_.skeletons = d;
			this.maxT = 1;
			this.maxN = 1;
			for (let skelName in d) {
				this.maxT = Math.max(this.maxT, d[skelName].T);
				this.maxN = Math.max(this.maxN, d[skelName].N);
			}
			sliderSelectedFrame.max = this.maxT-1;
			sliderSelectedIdx.max = this.maxN-1;
			this.curT = Math.min(this.curT, this.maxT);
			this.curN = Math.min(this.curN, this.maxN);
			sliderSelectedFrame.value = this.curT;
			sliderSelectedFrame.value = this.curN;
		}

		this.renderScene = function(ctx) {
			let cam = this.cam;
			let this_ = this;

			// Slow down render rate if inactive.
			// NOTE: Only issue here is that we must wait upto the longest-slept time before seeing new update!
			/*
			if ((new Date().getTime() - cam.lastMovement) > 12000) {
				this.sleepTime = 2000;
			} else if ((new Date().getTime() - cam.lastMovement) > 5000) {
				this.sleepTime = 1000;
			} else if ((new Date().getTime() - cam.lastMovement) > 2000) {
				this.sleepTime = 250;
			} else if ((new Date().getTime() - cam.lastMovement) > 1000) {
				this.sleepTime = 60;
			} else {
				this.sleepTime = 14;
			}
			*/

			this.setupScene({
				cam: cam
			}, function(ctx,props) {
				cam.step(ctx);

				this_.axes();
				for (let skel in this_.skeletons) {
					this_.skeletons[skel].draw(ctx, {timeIdx: this_.curT, showLabels: this_.showLabels});
				}
			});
		}

		get_skeletons();
}

window.addEventListener('load', () => {
	let container = document.getElementById("container");
	let regl = regl_({container: container, extensions: ['angle_instanced_arrays']});
	console.log(regl.limits)

	let app = new App(regl);

	// let skel = create_dummy_skeleton(regl);

	window.tick = regl.frame((ctx) => app.renderScene(ctx));
	// window.tick = setTimeout(function() {
	// });
	/*
	function draw_regl_cb() {
		let ctx = regl.contextState
		app.renderScene(ctx);
		// console.log(app.sleepTime);
		setTimeout(draw_regl_cb, app.sleepTime);
	}
	window.tick = setTimeout(draw_regl_cb);
	*/
});

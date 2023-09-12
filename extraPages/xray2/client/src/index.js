import regl_ from "regl";
import math from "./math3d.js";
import {create_dummy_skeleton, Skeleton} from "./skeleton.js";

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

function get_skeletons() {
	fetch('/getSkeletons')
		.then((response) => response.json())
		.then((d) => {
			setSkeletons(d)
		});

	// window.app.setSkeletons({
		// dummy: create_dummy_skeleton(window.app.regl)
	// });
}
function get_frame(i) {
}

function App(regl) {
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
		this.skeletons = {};


		this.setSkeletons = function(d) {
			this.skeletons = d;
		}

		this.renderScene = function(ctx) {
			let cam = this.cam;
			let this_ = this;

			this.setupScene({
				cam: cam
			}, function(ctx,props) {
				cam.step(ctx);
				this_.axes();

				for (let skel in this_.skeletons)
					this_.skeletons[skel].draw();
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

	let tick = regl.frame((ctx) => app.renderScene(ctx));
});

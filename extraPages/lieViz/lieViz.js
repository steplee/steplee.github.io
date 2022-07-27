const regl = require('regl')()
const mat4 = require('gl-mat4')

let normalBlend = {
		enable: true,
		func: {
			srcRGB: 'src alpha',
			srcAlpha: 1,
			dstRGB: 'one minus src alpha',
			dstAlpha: 1
		},
		equation: {
			rgb: 'add',
			alpha: 'add'
		},
	}

// Return [verts, inds] for a sphere in [-1,1]^3
function make_sphere() {
	let R = 20;
	let C = R*2;
	let verts = [];
	let inds = [];

	for (var i=0; i<R; i++) {
		for (var j=0; j<C; j++) {

			let u = (j/(C-1)) * Math.PI * 2;
			let v = (i/(R-1)) * Math.PI;
			verts.push(Math.sin(v)*Math.cos(u));
			verts.push(Math.sin(v)*Math.sin(u));
			verts.push(Math.cos(v));

			if (i<R-1 && j<C-1) {
				inds.push((i  )*C+(j  ));
				inds.push((i  )*C+(j+1));
				inds.push((i+1)*C+(j+1));

				inds.push((i+1)*C+(j+1));
				inds.push((i+1)*C+(j  ));
				inds.push((i  )*C+(j  ));
			}
		}
	}

	return [verts, new Uint16Array(inds)];
}

// Return [verts, inds] for a plane in [-1,1]^3
function make_plane() {
	let R = 10;
	let C = R;
	let verts = [];
	let inds = [];

	for (var i=0; i<R; i++) {
		for (var j=0; j<C; j++) {

			verts.push(2. * (i/R - .5));
			verts.push(2. * (j/R - .5));
			verts.push(0);

			if (i<R-1 && j<C-1) {
				inds.push((i  )*R+(j  ));
				inds.push((i  )*R+(j+1));
				inds.push((i+1)*R+(j+1));
				inds.push((i+1)*R+(j+1));
				inds.push((i+1)*R+(j  ));
				inds.push((i  )*R+(j  ));
			}
		}
	}

	return [verts, new Uint16Array(inds)];
}

// Calling regl() creates a new partially evaluated draw command
const drawTriangle = regl({

	// Shaders in regl are just strings.  You can use glslify or whatever you want
	// to define them.  No need to manually create shader objects.
	frag: `
	precision mediump float;
	uniform vec4 color;
	void main() {
		gl_FragColor = color;
	}`,

	vert: `
	precision mediump float;
	attribute vec2 position;
	void main() {
		gl_Position = vec4(position, 0, 1);
	}`,

	// Here we define the vertex attributes for the above shader
	attributes: {
		// regl.buffer creates a new array buffer object
		position: regl.buffer([
			[-2, -2],   // no need to flatten nested arrays, regl automatically
			[4, -2],    // unrolls them into a typedarray (default Float32)
			[4,  4]
		])
		// regl automatically infers sane defaults for the vertex attribute pointers
	},

	uniforms: {
		// This defines the color of the triangle to be a dynamic variable
		color: regl.prop('color')
	},

	// This tells regl the number of vertices to draw in this command
	count: 3
})

let [sv,si] = make_sphere()
console.log('inds',si)
const drawSphere = regl({

	frag: `
	precision mediump float;
	uniform vec4 color;
	varying vec3 v_pos;
	void main() {
		gl_FragColor = color;
		gl_FragColor.g += sin(v_pos.y*10.) * .5 + .5;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	attribute vec3 position;
	varying vec3 v_pos;
	void main() {
		vec4 p = mvp * vec4(position, 1.0);
		v_pos = p.xyz / p.w;
		gl_Position = p;
	}`,

	attributes: {
		position: sv
	},
	elements: regl.elements({primitive:'triangles', data: si}),

	uniforms: {
		color: regl.prop('color'),
		mvp: regl.prop('mvp')
	},

	cull: {enable: true},
	frontFace: 'cw',
	blend: normalBlend
})

let curveVerts1 = regl.buffer({
	usage: 'stream',
	type: 'float',
	length: 4*4*100
})

const drawCurve = regl({
	frag: `
	precision mediump float;
	uniform vec4 color;
	varying vec3 v_pos;
	varying float v_t;
	void main() {
		gl_FragColor = color;
		gl_FragColor.g += sin(v_pos.y*10.) * .5 + .5;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	attribute vec4 position_t;
	varying vec3 v_pos;
	varying float v_t;
	void main() {
		vec4 p = mvp * vec4(position_t.xyz, 1.0);

		v_pos = p.xyz / p.w;
		v_t = position_t.w;
		gl_Position = p;
	}`,

	attributes: {
		position_t: curveVerts1
	},
	primitive: 'line strip',
	count: 100,
	// elements: regl.elements({primitive:'points', count: 100}),

	uniforms: {
		color: regl.prop('color'),
		mvp: regl.prop('mvp')
	},
	blend: normalBlend
})

let arrowVerts = regl.buffer({
	usage: 'stream',
	type: 'float',
	data: new Float32Array([
		-1,0, 0,
		1,0, 0,
		1, 0, 0,
		-1, 0, 0,

		-2, 1, 0,
		2, 1, 0,
		0, 2, 0
	])
});
let arrowElements = regl.elements({
	primitive: 'triangles',
	data: [0,1,2, 2,3,0, 4,5,6]
	// data: [0,1,2, 2,3,0]
});
const drawArrow = regl({
	frag: `
	precision mediump float;
	uniform vec4 color;
	varying vec3 v_pos;
	void main() {
		gl_FragColor = color;
		gl_FragColor.g += 1.;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	uniform float tailLen;
	attribute vec3 position;
	varying vec3 v_pos;
	void main() {
		vec3 pp = position;
		vec4 p = mvp * vec4(pp, 1.0);

		v_pos = p.xyz / p.w;
		gl_Position = p;
	}`,

	attributes: {
		position: arrowVerts
	},
	elements: arrowElements,
	// elements: regl.elements({primitive:'points', count: 100}),

	cull: {enable:false},
	blend: { enable: true },

	uniforms: {
		color: regl.prop('color'),
		mvp: function(context, props, bid) {
			return props.mvp;
		}
	},
})

function norm3(a) {
	return Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}
function normalize3(a) {
	let l = norm3(a);
	return [a[0]/l, a[1]/l, a[2]/l];
}
function dist3(a,b) {
	let dx = b[0] - a[0];
	let dz = b[1] - a[1];
	let dy = b[2] - a[2];
	return Math.sqrt(dx*dx+dy*dy+dz*dz);
}
function drawArrowFromTo(a, b, color, mvp, view) {
	let verts = []
	let zplus = [view[2*4+0], view[2*4+1], view[2*4+2]]
	// let zplus = [view[0*4+1], view[1*4+1], view[2*4+1]]
	// let zplus = [view[1*4+0], view[1*4+1], view[1*4+2]]
	let eye = [view[3*4+0], view[3*4+1], view[3*4+2]]

	// Technically scale is dependent on FoV, but i'll ignore that
	let da = dist3(a,eye);
	let db = dist3(b,eye);
	let ab = dist3(a,b);
	let hs = .9;



	// It seems like we could do a projective transformation, like lookAt + a divideByZ
	// in order to get the verts without writing them.
	// But not exactly -- that would scale the head disproportionately.
	// So I'll scale/shift verts cpu side, but still put the scale/rotation into the MVP
	let w = .5;
	let f = .04 / da;
	let n = .04 / db;
	verts = new Float32Array([
		-1*f*w,0, ab*1-0,
		 1*f*w,0, ab*1-0,
		 1*n*w,0, (1-hs)*ab,
		-1*n*w,0, (1-hs)*ab,

		-2*n,0, (1-hs)*ab,
		 2*n,0, (1-hs)*ab,
		 0,0, 1-1
	])

	let new_mvp = mat4.create();
	// mat4.lookAt(new_mvp, b, a, zplus)
	let ma = [-a[0],-a[1],-a[2]];
	let mb = [-b[0],-b[1],-b[2]];
	mat4.lookAt(new_mvp, mb,ma, zplus)
	mat4.multiply(new_mvp, mvp, new_mvp);


	arrowVerts.subdata(verts)

	drawArrow({mvp: new_mvp, color: color})
}

let it = regl.frame((ctx) => {
	// console.log("render @",time)
	let time = ctx.time;
	let w = ctx.viewportWidth, h = ctx.viewportHeight;

	let proj = mat4.create()
	let zn = .0001
	// mat4.frustum(proj, -.5, .5, -.5, .5, zn, 50)
	mat4.frustum(proj, -.5*zn*w/h, .5*zn*w/h, -.5*zn, .5*zn, zn, 50)

	let view = mat4.create()
	// mat4.lookAt(view, [2,2,15.2], [0,0,0], [0,0,1])
	mat4.lookAt(view, [2,2,.0], [0,0,0], [0,0,1])
	mat4.rotateZ(view,view,time*.5);

	let mvp = mat4.create()
	mat4.multiply(mvp, proj, view);
	// console.log("render @",mvp)

	// clear contents of the drawing buffer
	regl.clear({
		color: [0, 0, 0, 1],
		depth: 1
	})


	if (time < .1) {
	let newVerts = []
	for (var i=0; i<100; i++) {
		// let a = 1. + Math.exp(-Math.cos(Math.PI*(time - i))*3.)
		let a = .3 + Math.exp((.5+.5*Math.cos(Math.PI*(time - 9.*i/100))) * -20.) * .3
		newVerts.push((i/100)*a*Math.cos(2*3.141*i/100))
		newVerts.push((i/100)*a*Math.sin(2*3.141*i/100))
		// newVerts.push(i/100-.5)
		newVerts.push(i/100)
		// newVerts.push(0/100)
		// newVerts.push(0)
		newVerts.push(i/100)
	}
	// console.log(newVerts)
	curveVerts1.subdata(newVerts)
	}

	drawCurve({
		mvp: mvp,
		color: [ Math.cos(time * 0.1), Math.sin(time * 0.8), Math.cos(time * 0.3), 1 ]
	})

	drawArrowFromTo(
		[-0,.6,.8], [.0,.8,0.0],
		[ Math.cos(time * 0.1), Math.sin(time * 0.8), Math.cos(time * 0.3), 1 ], mvp, view
	)

	drawSphere({ mvp: mvp, color: [ Math.cos(time * 0.1), Math.sin(time * 0.8), Math.cos(time * 0.3), .1 ] })

	// it.cancel()
})

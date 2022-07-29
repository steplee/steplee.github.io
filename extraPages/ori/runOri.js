const regl = require('regl')()
const mat4 = require('gl-mat4')

/*
function hello() {
	console.log(`array has ${window.simData.length} entries`)
}

window.onload = () => {
	hello()
}
*/

function multiplyMatVec(out, m, v) {
	// for (let i=0; i<3; i++) out[i] = m[i*4+3];
	for (let i=0; i<3; i++) out[i] = m[3*4+i];

	for (let i=0; i<3; i++) {
		for (let j=0; j<3; j++) {
			out[i] += m[j*4+i] * v[j];
		}
	}
}

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

function getYaw(a) {
	return new Float32Array([
		Math.cos(a),  0, Math.sin(a), 0,
		0,            1, 0,           0,
		-Math.sin(a), 0, Math.cos(a), 0,
		0,            0, 0,           1])
}
function getPitch(a) {
	return new Float32Array([
		1, 0,            0,           0,
		0, Math.cos(a),  Math.sin(a), 0,
		0, -Math.sin(a), Math.cos(a), 0,
		0, 0,            0,           1])
}
function getRoll(a) {
	return new Float32Array([
		Math.cos(a),  Math.sin(a), 0, 0,
		-Math.sin(a), Math.cos(a), 0, 0,
		0,            0,           1, 0,
		0,            0,           0, 1])
}
function cross(a,b) {
	return [
		-a[2]*b[1] + a[1]*b[2],
		 a[2]*b[0] - a[0]*b[2],
		-a[1]*b[0] + a[0]*b[1]];
}

function make_plane() {
	let offset = [0,0,0];
	// let offset = [0,0,1];
	// let offset = [0,0,-.5];
	let verts = [
		0, 0, 1.8,
		0, .5, .1,
		-1, 0, 0,
		1, 0, 0,
		0,0,0,
		-.5,0,.4,
		.5,0,.4].map((x,i) => x + offset[Math.floor(i%3)])
	let inds = [
		// bottom
		0,5,4,
		6,0,4,
		5,2,4,
		3,6,4,
		// top
		0,1,5,
		1,0,6,
		1,4,2,
		4,1,3,
		5,1,2,
		1,6,3 ];
	let normals = new Float32Array(verts.length)
	for (var i=0; i<inds.length/3; i++) {
		let ia = inds[i*3+0];
		let ib = inds[i*3+1];
		let ic = inds[i*3+2];
		let ca = [verts[ic*3+0] - verts[ia*3+0], verts[ic*3+1] - verts[ia*3+1], verts[ic*3+2] - verts[ia*3+2]]
		let ba = [verts[ib*3+0] - verts[ia*3+0], verts[ib*3+1] - verts[ia*3+1], verts[ib*3+2] - verts[ia*3+2]]
		let n = cross(ca,ba)
		for (var j of [ia,ib,ic]) {
			normals[j*3+0] += n[0]
			normals[j*3+1] += n[1]
			normals[j*3+2] += n[2]
		}
	}
	for (var i=0; i<normals.length/3; i++) {
		let n = Math.sqrt( .0000000001 + normals[i*3+0]*normals[i*3+0] + normals[i*3+1]*normals[i*3+1] + normals[i*3+2]*normals[i*3+2]);
		normals[i*3+0] /= n
		normals[i*3+1] /= n
		normals[i*3+2] /= n
	}
	let tip = new Float32Array([verts[0], verts[1], verts[2]])
	return [new Float32Array(verts), new Uint16Array(inds), new Float32Array(normals), tip];
}

function get_ring_verts() {
	let radius = 1
	let verts = Array(32*3).fill(0).map((a,i) => {
		let j = Math.PI * 2 * Math.floor(i / 3 + .1);
		return radius * (i % 3 == 0 ? Math.cos(j/31) : i % 3 == 1 ? Math.sin(j/31) : 0.)
	});
	return new Float32Array(verts)
}
function make_sector() {
	let verts = [], inds = []
	for (var y=0; y<2; y++) {
		for (var x=0; x<32; x++) {
			verts.push(x/31);
			verts.push(y);
		}
	}
	for (var x=0; x<32-1; x++) {
		inds.push((0+0)*32+x  )
		inds.push((0+0)*32+x+1)
		inds.push((0+1)*32+x+1)

		inds.push((0+1)*32+x+1)
		inds.push((0+1)*32+x  )
		inds.push((0+0)*32+x  )
	}
	return [new Float32Array(verts), new Uint16Array(inds)];
}

let [plane_v,plane_i, plane_n, planeTipPos] = make_plane()
const drawPlane = regl({
	frag: `
	precision mediump float;
	uniform vec4 color;
	uniform mat4 mvp;
	varying vec3 v_pos;
	varying vec3 v_nrl;
	void main() {
		vec4 c = color;
		c.rgb *= .5 + .3 * clamp(dot(v_nrl, normalize(vec3(mvp[0][2],mvp[1][2],mvp[2][2]))),0.,1.);
		gl_FragColor = c;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	attribute vec3 position;
	attribute vec3 normal;
	varying vec3 v_pos;
	varying vec3 v_nrl;
	void main() {
		vec3 pp = position;
		// pp.z += 1.;
		pp.x *= .6;
		vec4 p = mvp * vec4(pp, 1.0);
		v_pos = p.xyz / p.w;
		v_nrl = normal;
		gl_Position = p;
	}`,

	attributes: {
		position: plane_v,
		normal: plane_n,
	},
	elements: regl.elements({primitive:'triangles', data: plane_i}),

	uniforms: {
		color: regl.prop('color'),
		mvp: regl.prop('mvp')
	},

	cull: {enable: true},
	// cull: {enable: true},
	// frontFace: 'cw',
	blend: normalBlend
})

let TRAIL_N = 256
let drawTrail = regl({
	frag: `
	precision mediump float;
	varying vec4 v_color;
	void main() {
		vec4 c = v_color;
		gl_FragColor = c;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	uniform vec4 color;
	uniform float time;
	attribute vec4 position_t;
	varying vec4 v_color;
	void main() {
		vec3 pp = position_t.xyz;
		vec4 p = mvp * vec4(pp, 1.0);
		gl_Position = p;

		float vtime = position_t.w;
		v_color = color;
		// v_color.rgba *= (1. - clamp(10.*(vtime - time), 0., 1.));
		v_color.a *= 1. - mix(3.*(time-vtime), 1., 0.);
		// v_color.rgba *= time;
	}`,
	attributes: { position_t: regl.prop('trailBuffer') },
	uniforms: { mvp: regl.prop('mvp'), color: regl.prop('color'), time: function(c) { return c.time*.1 } },

	offset: regl.prop('offset'),
	primitive: 'line strip',
	count: regl.prop('count'),
	cull: {enable: false},
	blend: normalBlend
})
function renderPlaneAndTrail(mvp, model, time, trailMeta) {
	let planeMvp = mat4.create();
	mat4.multiply(planeMvp, mvp, model);
	drawPlane({ mvp: planeMvp, color: [ 1,1,1,.7] })


	let tip = new Float32Array(4)
	multiplyMatVec(tip, model, planeTipPos)
	tip[3] = time * .1

	// let tip = [Math.cos(trail.idx/1000), Math.sin(trail.idx/1000), 0]
	trailMeta.trailBuffer.subdata(tip, trailMeta.idx*4*4)
	// Render [0, i]
	drawTrail({ mvp: mvp, color: [ 1,1,1,.7], offset: 0, count: trailMeta.idx, trailBuffer: trailMeta.trailBuffer })
	// Render [i+1, n]
	if (trailMeta.total > trailMeta.idx) {
		drawTrail({ mvp: mvp, color: [ 1,1,1,.7], offset: trailMeta.idx+1, count: (trailMeta.total-trailMeta.idx-1), trailBuffer: trailMeta.trailBuffer})
		// console.log(' - render', 0, '->', trailMeta.idx, 'AND', trailMeta.idx, '->', trailMeta.total)
	}

	trailMeta.idx = trailMeta.idx + 1
	if (trailMeta.idx >= TRAIL_N) trailMeta.idx = 0
	if (trailMeta.total < TRAIL_N) trailMeta.total += 1

}

let [sector_v, sector_i] = make_sector()
const drawAngleSingleRing = regl({
	frag: `
	precision mediump float;
	uniform vec4 color;
	uniform mat4 mvp;
	void main() {
		gl_FragColor = color;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	attribute vec3 position;
	void main() {
		vec4 p = mvp * vec4(position, 1.0);
		gl_Position = p;
	}`,

	attributes: { position: get_ring_verts() },
	primitive: 'line strip',
	count: 32,
	uniforms: {
		color: regl.prop('color'),
		mvp: regl.prop('mvp'),
	},
	cull: { enable: false },
	blend: normalBlend
})
const drawAngleSingleSector = regl({
	frag: `
	precision mediump float;
	uniform vec4 color;
	uniform mat4 mvp;
	void main() {
		gl_FragColor = color;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	uniform float angle;
	uniform float sectorSize;
	uniform float sectorOffset;
	attribute vec2 position;
	void main() {
		float theta  = position.x * angle;
		float radius = position.y * sectorSize + sectorOffset;
		vec3 pp = vec3(cos(theta), sin(theta), 0.) * radius;
		vec4 p = mvp * vec4(pp, 1.0);
		gl_Position = p;
	}`,

	attributes: { position: sector_v },
	elements: regl.elements({primitive:'triangles', data: sector_i}),
	count: 31*3*2,
	uniforms: {
		color: regl.prop('color'),
		mvp: regl.prop('mvp'),
		angle: regl.prop('angle'),
		sectorSize: regl.prop('sectorSize'),
		sectorOffset: regl.prop('sectorOffset'),
	},
	cull: { enable: false },
	blend: normalBlend
})

function drawAngleSingle(mvp, angle, color, sectorOffset=.1, sectorSize=.8) {
	drawAngleSingleRing({mvp: mvp, color:[color[0], color[1], color[2], color[3]*.5]})
	drawAngleSingleSector({mvp: mvp, angle: angle, color:color, sectorOffset:sectorOffset, sectorSize:sectorSize})
}
function drawAngleTriple(mvp, model, angles, colors) {
	// var mvp = mat4.clone(mvp)

	// Note: column-major
	// this could be done better by swizzling the pos in the shader...
	let modelMatrices = [
		 // Yaw
		 new Float32Array([
			 1,0,0,0,
			 0,1,0,0,
			 0,0,0,0,
			 0,0,0,1]),
		 // Pitch
		 new Float32Array([
			 0,1,0,0,
			 0,0,1,0,
			 0,0,0,0,
			 0,0,0,1]),
		 // Roll
		 new Float32Array([
			 1,0,0,0,
			 0,0,1,0,
			 0,0,0,0,
			 0,0,0,1]),
	]

	var sectorOffset = .1
	var sectorSize = .29;

	for (var i=angles.length-1; i>=0; i--) {

		let angle = angles[i], color = colors[i];

		let localMvp = mat4.create();
		mat4.multiply(localMvp, mvp, model);
		mat4.multiply(localMvp, localMvp, modelMatrices[i]);
		drawAngleSingle(localMvp, angle, color, sectorOffset, sectorSize);
		sectorOffset += sectorSize

		// Note: R is not applied until AFTER render this one
		// let R = i == 0 ? getYaw(angle) : i == 1 ? getPitch(angle) : getRoll(angle)
		let R = i == 0 ? getRoll(angle) : i == 1 ? getPitch(angle) : getYaw(angle)
		// Note: @view is modified in place!
		mat4.multiply(model, model, R)
	}
}

let worldFromPlatform = new Float32Array([
	1, 0, 0, 0,
	0, 0, -1, 0,
	0, 1, 0, 0,
	0, 0, 0, 1 ])

let g_trailMeta = {
	idx: 0,
	total: 0,
	trailBuffer: regl.buffer({
		usage: 'dynamic',
		data: new Float32Array(TRAIL_N*4),
		type: 'float32'
	})}

function Compartment(vport) {
	this.viewport = vport
	this.trailMeta = {
		idx: 0,
		total: 0,
		trailBuffer: regl.buffer({
			usage: 'dynamic',
			data: new Float32Array(TRAIL_N*4),
			type: 'float32'
		})}
}

// I think this could be a scoped command, but easier
// to just use a js func
Compartment.prototype.render = function(ctx, view) {
	let w = ctx.viewportWidth, h = ctx.viewportHeight;
	// let w = ctx.viewportWidth*this.viewport.w, h = ctx.viewportHeight*this.viewport.h;
	// console.log(this.viewport.w, this.viewport.h)

	console.log(w,h)
	let proj = mat4.create()
	let zn = .0001
	let f = .3
	mat4.frustum(proj, -f*zn*w/h, f*zn*w/h, -f*zn, f*zn, zn, 50)

	let model = mat4.create()
	mat4.multiply(view, view, worldFromPlatform);
	let mvp = mat4.create()
	mat4.multiply(mvp, proj, view);

	let time = ctx.time
	let t = time * 5.
	let angles = [
		Math.sin(t*.979),
		.8*Math.sin(t*.633),
		t
	]

	let colors = [  [0,0,1,.5],
					[0,1,0,.5],
					[1,0,0,.5]]

	// drawAngleSingle(mvp, angle1, [.2,.2,1,.2])
	drawAngleTriple(mvp, model, angles.map(a=>a%(2*Math.PI)), colors)

	renderPlaneAndTrail(mvp, model, time, g_trailMeta)

}
let comp1 = new Compartment()
	// viewport: regl.context('vport'),
	// viewport: regl.prop('vport')
	// context: { viewport: vport1 },

let it = regl.frame((ctx) => {

	regl.clear({color:[0,0,0,1], depth:1})

	let view = mat4.create()
	// Start South and Up, XYZ ~ ESU
	mat4.lookAt(view, [0,-9,-4.0], [0,0,0], [0,0,-1])
	comp1.render(ctx, view)

	// it.cancel()
})

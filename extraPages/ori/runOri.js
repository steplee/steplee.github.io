const regl = require('regl')()
const mat4 = require('gl-mat4')

// ----------------------------------------
//      JSON Data Retrieval
// ----------------------------------------

/*
 * Retrieve data from simAndFilterData.json
 * This contains the simulation ground truth data, as well as filter outputs.
 */
function handleDataJson(data) {
	console.log(' - Received data', data)
	setupSceneWithData(data)
}
async function queueLoadJsonData() {
	console.log(' - Fetching Json Data')
	let res = await fetch('simAndFilterData.json',
		{
			// credentials: 'same-origin'
		})
	let data = await res.json()
	handleDataJson(data)
}


// ----------------------------------------
//      Mouse Input & Camera class
// ----------------------------------------

document.myMouse = {
	dx: 0, dy: 0,
	lastTime: 0,
	leftClicked: false,
	rightClicked: false,
}
function handleMouseMovement(e) {
	document.myMouse.dx = e.movementX;
	document.myMouse.dy = e.movementY;
}
function handleMouseDown(e) {
	if (e.button == 0) document.myMouse.leftClicked = true
	if (e.button == 1) document.myMouse.rightClicked = true
}
function handleMouseUp(e) {
	if (e.button == 0) document.myMouse.leftClicked = false
	if (e.button == 1) document.myMouse.rightClicked = false
}
document.addEventListener('mousemove', handleMouseMovement);
document.addEventListener('mousedown', handleMouseDown);
document.addEventListener('mouseup', handleMouseUp);

class CameraViewOnly {
	constructor() {
		this.eye = [0,-9,-4]
		this.target = [0,0,0]
		this.up = [0,0,-1]
		this.eyeDist = norm(this.eye)
	}

	stepAndComputeView() {

		let dx = document.myMouse.dx, dy = document.myMouse.dy
		let leftDown = document.myMouse.leftClicked
		if (leftDown && ((dx != 0) || (dy != 0))) {
			const e = this.eye

			// You should apply rodrigues formula, but since dx and dy are small,
			// treat as infitisemal and renormalize at end
			//
			// Note: the 'yaw' must be done along current X+, which is the longer expression
			// Note: axes are sort of messed up, have not yet applied worldFromPlatform
			let add_dx = cross(e, [0, 0, dx*.01])
			let add_dy = cross(e, cross([e[0]*dy*.01/this.eyeDist, e[1]*dy*.01/this.eyeDist, e[2]*dy*.01/this.eyeDist], this.up))
			let newEye = [
				e[0] + add_dx[0] + add_dy[0],
				e[1] + add_dx[1] + add_dy[1],
				e[2] + add_dx[2] + add_dy[2]]
			let n = norm(newEye)
			this.eye = [ newEye[0]/n * this.eyeDist, newEye[1]/n * this.eyeDist, newEye[2]/n * this.eyeDist]
		}

		let view = mat4.create()
		mat4.lookAt(view, this.eye, this.target, this.up)

		document.myMouse.dx = 0
		document.myMouse.dy = 0

		return view
	}
}
document.myCamera = new CameraViewOnly()


// ----------------------------------------
//      Math Functions
// ----------------------------------------

function multiplyMatVec(out, m, v) {
	// for (let i=0; i<3; i++) out[i] = m[i*4+3];
	for (let i=0; i<3; i++) out[i] = m[3*4+i];

	for (let i=0; i<3; i++) {
		for (let j=0; j<3; j++) {
			out[i] += m[j*4+i] * v[j];
		}
	}
}

function norm(x) {
	return Math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
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

// ----------------------------------------
//      Geometry & Graphics Utilities
// ----------------------------------------

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
		.5,0,.4]
		.map((x,i) => x + offset[Math.floor(i%3)])
		.map((x,i) => (i%3) == 1 ? -x : x) // whoops, had to flip
	let inds_ = [
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

	// let inds = inds_
	let inds = inds_.map((x,i) => (i%3==0)?inds_[i+1]:(i%3==1)?inds_[i-1]:x) // whoops

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

const drawGizmo = regl({
	frag: `
	precision mediump float;
	varying vec3 v_color;
	void main() {
		vec4 c = vec4(v_color, .5);
		gl_FragColor = c;
	}`,

	vert: `
	precision mediump float;
	uniform mat4 mvp;
	attribute vec3 color;
	attribute vec3 position;
	varying vec3 v_color;
	void main() {
		vec4 p = mvp * vec4(position.xyz, 1.0);
		gl_Position = p;
		v_color = color;
	}`,
	attributes: { position: [
		// [0,0,0], [1,0,0],
		// [0,0,0], [0,1,0],
		// [0,0,0], [0,0,1] ],
		[1,0,0], [2,0,0],
		[0,1,0], [0,2,0],
		[0,0,1], [0,0,2] ],
	color: [
		[1,0,0], [1,0,0],
		[0,1,0], [0,1,0],
		[0,0,1], [0,0,1] ] },
	uniforms: { mvp: regl.prop('mvp') },

	offset: 0,
	primitive: 'lines',
	count: 6,
	blend: normalBlend
})

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
	// frontFace: 'ccw',
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
	uniforms: { mvp: regl.prop('mvp'), color: regl.prop('color'), time: regl.prop('time') },

	offset: regl.prop('offset'),
	primitive: 'line strip',
	count: regl.prop('count'),
	cull: {enable: false},
	blend: normalBlend
})
function renderPlaneAndTrail(mvp, model, time_, trailMeta, pushTrail) {
	let planeMvp = mat4.create();
	mat4.multiply(planeMvp, mvp, model);
	drawPlane({ mvp: planeMvp, color: [ 1,1,1,.7] })

	let time = time_ * .1


	if (pushTrail) {
		let tip = new Float32Array(4)
		multiplyMatVec(tip, model, planeTipPos)
		tip[3] = time

		// let tip = [Math.cos(trail.idx/1000), Math.sin(trail.idx/1000), 0]
		trailMeta.trailBuffer.subdata(tip, trailMeta.idx*4*4)
	}

	// Render [0, i]
	drawTrail({ mvp: mvp, color: [ 1,1,1,.7], offset: 0, count: trailMeta.idx, trailBuffer: trailMeta.trailBuffer, time:time })
	// Render [i+1, n]
	if (trailMeta.total > trailMeta.idx) {
		drawTrail({ mvp: mvp, color: [ 1,1,1,.7], offset: trailMeta.idx+1, count: (trailMeta.total-trailMeta.idx-1), trailBuffer: trailMeta.trailBuffer, time:time})
		// console.log(' - render', 0, '->', trailMeta.idx, 'AND', trailMeta.idx, '->', trailMeta.total)
	}

	if (pushTrail) {
		trailMeta.idx = trailMeta.idx + 1
		if (trailMeta.idx >= TRAIL_N) trailMeta.idx = 0
		if (trailMeta.total < TRAIL_N) trailMeta.total += 1
	}

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
		 // Roll
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
		 // Yaw
		 new Float32Array([
			 0,0,1,0,
			 -1,0,0,0,
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


function genTrailMeta() {
	return {
		idx: 0,
		total: 0,
		trailBuffer: regl.buffer({
			usage: 'dynamic',
			data: new Float32Array(TRAIL_N*4),
			type: 'float32'
		})}
}

const worldFromPlatform = new Float32Array([
	-1, 0, 0, 0,
	0, 0, 1, 0,
	0, 1, 0, 0,
	// off[0], off[1], off[2], 1 ])
	0,0,0,1])

function Compartment(vport, name) {
	this.vport = vport
	this.trailMeta = genTrailMeta()
	this.name = name
}
Compartment.prototype.draw = regl({
	context: {
		hello: 1
	}
})
function renderCompartment(c, view, angles, pushTrail=true) {
	c.draw({}, function(ctx) {

		let w = ctx.viewportWidth, h = ctx.viewportHeight;
		// let w = ctx.viewportWidth*this.viewport.w, h = ctx.viewportHeight*this.viewport.h;
		// console.log(this.viewport.w, this.viewport.h)

		let proj = mat4.create()
		let zn = .003
		let f = .3
		// mat4.frustum(proj, -f*zn*w/h, f*zn*w/h, -f*zn, f*zn, zn, 50)
		// mat4.frustum(proj, -f*zn*w/h, f*zn*w/h, -f*zn, f*zn, zn, 50)
		let x0 = (f*zn*w/h * (-1 + c.vport[0])) * c.vport[2];
		let x1 = (f*zn*w/h * ( 1 + c.vport[0])) * c.vport[2];
		let y0 = (f*zn     * (-1 + c.vport[1])) * c.vport[3];
		let y1 = (f*zn     * ( 1 + c.vport[1])) * c.vport[3];
		mat4.frustum(proj, x0,x1, y0,y1, zn, 20)


		let model = mat4.create()
		let mvp = mat4.create()
		mat4.multiply(mvp, proj, view);

		drawGizmo({mvp:mvp})

		let rpyColors = [[0,0,1,.5],
						 [1,0,0,.5],
						 [0,1,0,.5]]

		// drawAngleSingle(mvp, angle1, [.2,.2,1,.2])
		drawAngleTriple(mvp, model, angles.map(a=>a%(2*Math.PI)), rpyColors)

		let time = ctx.time
		renderPlaneAndTrail(mvp, model, time, c.trailMeta, pushTrail)
	})
}


function setupSceneTest() {
	let comp1 = new Compartment([-.43,0,1.,1.], [0,0,0])
	let comp2 = new Compartment([.43,0,1.,1.], [0,0,0])
	let it = regl.frame((ctx) => {

		regl.clear({color:[0,0,0,1], depth:1});

		/*
			let view = mat4.create()
		let tt = Math.sin(ctx.time) * .3
		// Start South and Up, XYZ ~ ESU
		// mat4.lookAt(view, [0,-9,-4.0], [0,0,0], [0,0,-1])
		mat4.lookAt(view, [-9*Math.sin(tt),-9*Math.cos(tt),-4.0], [0,0,0], [0,0,-1])
		mat4.multiply(view, view, worldFromPlatform);
		*/

		let view = document.myCamera.stepAndComputeView();
		mat4.multiply(view, view, worldFromPlatform);


		let time = ctx.time;
		var t = time * 2.;
		var angles = [ Math.sin(t*.979)+Math.cos(t*.1), .8*Math.sin(t*.633)*Math.cos(t*.7), t ];

		renderCompartment(comp1, view, angles);

		t = time * 5.;
		angles = [ 0,t*.2, t ];

		renderCompartment(comp2, view, angles, pushTrail=j!=last_j);
		last_j = j
	})
}

class PlaybackScene {

	constructor(data) {
		this.data = data

		let trueComp = new Compartment([0,0,1,1], "");
		let filterComp = new Compartment([0,0,1,1], "RpyFilter");
		// let filterComps = [];

		let last_j = -1
		let n = data['outputs']['RpyApprox']['rpysShape'][0]

		let it = regl.frame((ctx) => {
			regl.clear({color:[0,0,0,1], depth:1});

			let view = document.myCamera.stepAndComputeView();
			mat4.multiply(view, view, worldFromPlatform);

			let time = ctx.time;
			var t = time * 2.;

			const TIME_SCALE = 60;
			let j = Math.round(Math.min(time * TIME_SCALE, n-1))

			let angles1 = data['outputs']['RpyApprox']['rpys'][j]
			renderCompartment(filterComp, view, angles1, j!=last_j);

			let angles2 = data['simRpys'][j];
			renderCompartment(trueComp, view, angles2, j!=last_j);
			if (j != last_j) {
				let dx = (angles1[0] - angles2[0]); dx = 2*Math.PI-dx < dx ? 2*Math.PI-dx : dx
				let dy = (angles1[1] - angles2[1]); dy = 2*Math.PI-dy < dy ? 2*Math.PI-dy : dy
				let dz = (angles1[2] - angles2[2]); dz = 2*Math.PI-dz < dz ? 2*Math.PI-dz : dz
				console.log(' - rpy error', Math.sqrt(dx*dx+dy*dy+dz*dz))
			}

			last_j = j
		});
	}

	run() {
	}
}

function setupSceneWithData(data) {
	// Create two components for each filter,
	// one displaying the truth, one the filter output

	var lastTime = 0;

	console.log(data)
	for (const [filterName,vals] of Object.entries(data['outputs'])) {
		for (const [valName,valData] of Object.entries(vals)) {
			console.log(` - Filter "${filterName}" has values named "${valName}"`)
		}
	}

	window.scene = new PlaybackScene(data);
	window.scene.run();

}

//setupSceneTest()

// Call the load, which eventually sets up scene using setupSceneWithData
queueLoadJsonData();

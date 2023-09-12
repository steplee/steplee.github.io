
//
// NOTE: Mostly copied & edited from my row-major version.
// There may be bugs.
//

// ---------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------

function eye() {
	return new Float32Array([
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1]);
}

function mtv(A, x) {
	return new Float32Array([
		A[0*4+0]*x[0] + A[0*4+1]*x[1] + A[0*4+2]*x[2] + A[0*4+3]*x[3],
		A[1*4+0]*x[0] + A[1*4+1]*x[1] + A[1*4+2]*x[2] + A[1*4+3]*x[3],
		A[2*4+0]*x[0] + A[2*4+1]*x[1] + A[2*4+2]*x[2] + A[2*4+3]*x[3],
		A[3*4+0]*x[0] + A[3*4+1]*x[1] + A[3*4+2]*x[2] + A[3*4+3]*x[3]]);
}

// Transpose the top-left 3x3 corner of A and multiply the 3-vector x
function topLeft_mtv_3(A, x) {
	return new Float32Array([
		A[0*4+0]*x[0] + A[0*4+1]*x[1] + A[0*4+2]*x[2],
		A[1*4+0]*x[0] + A[1*4+1]*x[1] + A[1*4+2]*x[2],
		A[2*4+0]*x[0] + A[2*4+1]*x[1] + A[2*4+2]*x[2]]);
}

function mv(A, x) {
	return new Float32Array([
		A[0*4+0]*x[0] + A[1*4+0]*x[1] + A[2*4+0]*x[2] + A[3*4+0]*x[3],
		A[0*4+1]*x[0] + A[1*4+1]*x[1] + A[2*4+1]*x[2] + A[3*4+1]*x[3],
		A[0*4+2]*x[0] + A[1*4+2]*x[1] + A[2*4+2]*x[2] + A[3*4+2]*x[3],
		A[0*4+3]*x[0] + A[1*4+3]*x[1] + A[2*4+3]*x[2] + A[3*4+3]*x[3]]);
}

function mm(A, B) {
	let C = new Float32Array(16);
	for (let i=0; i<4; i++)
		for (let j=0; j<4; j++)
			for (let k=0; k<4; k++)
				C[j*4+i] += A[k*4+i]*B[j*4+k];
	return C;
}

function inverseSE3(A) {
	const x = -(A[0*4+0]*A[3*4+0] + A[0*4+1]*A[3*4+1] + A[0*4+2]*A[3*4+2]);
	const y = -(A[1*4+0]*A[3*4+0] + A[1*4+1]*A[3*4+1] + A[1*4+2]*A[3*4+2]);
	const z = -(A[2*4+0]*A[3*4+0] + A[2*4+1]*A[3*4+1] + A[2*4+2]*A[3*4+2]);
	return new Float32Array([
		// A[0+4*0], A[1+4*0], A[2+4*0], 0,
		// A[0+4*1], A[1+4*1], A[2+4*1], 0,
		// A[0+4*2], A[1+4*2], A[2+4*2], 0,
		A[0*4+0], A[1*4+0], A[2*4+0], 0,
		A[0*4+1], A[1*4+1], A[2*4+1], 0,
		A[0*4+2], A[1*4+2], A[2*4+2], 0,
		x,y,z,1]);
}

function inverseGeneral(A) {
}
function inverseProj(A) {
	// Use sympy: https://live.sympy.org/
	//
	// a,b,c,d,e,f = symbols('a b c d e f')
	// P = Matrix([ [a,0,b,0], [0,c,d,0], [0,0,e,f], [0,0,-1,0]])
	// P.inv()
	//
	const a = A[0*4+0];
	const b = A[2*4+0];
	const c = A[1*4+1];
	const d = A[2*4+1];
	const e = A[2*4+2];
	const f = A[3*4+2];
	return new Float32Array([
		// 1/a, 0, 0, b/a,
		// 0, 1/c, 0, d/c,
		// 0, 0, 0, -1,
		// 0, 0, 1/f, e/f
		1/a, 0, 0, 0,
		0, 1/c, 0, 0,
		0, 0, 0, 1/f,
		b/a, d/c, -1, e/f
	]);
}

function normalized(x) {
	const n = Math.sqrt(x.map(e=>e*e).reduce((a,b)=>a+b, 0));
	return x.slice().map(a=>a/n);
}
function norm(x) {
	return Math.sqrt(x.map(e=>e*e).reduce((a,b)=>a+b, 0));
}

function cross(x,y) {
	return new Float32Array([
		-x[2]*y[1] + x[1]*y[2],
		 x[2]*y[0] - x[0]*y[2],
		-x[1]*y[0] + x[0]*y[1]])
}

function crossMatrix(x) {
	return new Float32Array([
		0,x[2],-x[1],
		-x[2],0, x[0],
		 x[1],-x[0],0]);
}

function transpose(A) {
	// return A;
	return new Float32Array([
		A[0*4+0], A[1*4+0], A[2*4+0], A[3*4+0],
		A[0*4+1], A[1*4+1], A[2*4+1], A[3*4+1],
		A[0*4+2], A[1*4+2], A[2*4+2], A[3*4+2],
		A[0*4+3], A[1*4+3], A[2*4+3], A[3*4+3]]);
}

exports.eye = eye;
exports.mv = mv;
exports.mtv = mtv;
exports.mm = mm;
exports.inverseSE3 = inverseSE3;
exports.inverseGeneral = inverseGeneral;
exports.inverseProj = inverseProj;
exports.normalized = normalized;
exports.cross = cross;
exports.crossMatrix = crossMatrix;
exports.transpose = transpose;

// ---------------------------------------------------------------
// Graphics
// ---------------------------------------------------------------

function lookAt(eye, target, up) {
	const diff = [target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]];
	let f = normalized(diff);
	let r = normalized(cross(up, f));
	let u = normalized(cross(f,r));
	r = cross(u,f);
	return inverseSE3(new Float32Array([
		// r[0], u[0], f[0], 0,
		// r[1], u[1], f[1], 0,
		// r[2], u[2], f[2], 0,
		r[0], r[1], r[2], 0,
		u[0], u[1], u[2], 0,
		f[0], f[1], f[2], 0,
		eye[0], eye[1], eye[2], 1]));
}

function frustum(n,f, l,r, t,b) {
	return transpose(
		new Float32Array([
		2*n/(r-l), 0, (r+l)/(r-l), 0,
		0, 2*n/(t-b), (t+b)/(t-b), 0,
		0, 0,  (f+n)/(f-n),  -2*f*n/(f-n),
		0,0,1,0]));
}
function frustumFromIntrin(n,f, intrin) {
}

exports.lookAt = lookAt;
exports.frustum = frustum;
exports.frustumFromIntrin = frustumFromIntrin;

// ---------------------------------------------------------------
// Camera Control
// ---------------------------------------------------------------

// To be used with https://github.com/mikolalysenko/mouse-change/blob/master/mouse-listen.js
function FreeCamera(element) {
	this.fov = 45;
	this.q = new Float32Array([1,0,0,0]);
	this.t = new Float32Array([0,0,-1.1]);
	this.v = new Float32Array([0,0,0]);
	this.computeMatrices({viewportWidth:1, viewportHeight:1});

	this.lastX = undefined;
	this.lastY = undefined;
	this.lastTime = undefined;
	this.leftDown = false;
	this.rghtDown = false;
	this.dx = 0;
	this.dy = 0;
	this.shift = false;
	this.alt = false;
	this.keys = [];

	// element.addEventListener('mousemove', this.handleMouseMove)
	// element.addEventListener('mousedown', this.handleMouseDown)
	// element.addEventListener('mouseup', this.handleMouseUp)
	element.addEventListener('mouseup', this.handleMouse.bind(this))
	element.addEventListener('mousedown', this.handleMouse.bind(this))
	element.addEventListener('mousemove', this.handleMouse.bind(this))
	element.addEventListener('mouseleave', this.handleClear.bind(this))
	element.addEventListener('mouseenter', this.handleClear.bind(this))
	element.addEventListener('mouseover', this.handleClear.bind(this))
	element.addEventListener('mouseout', this.handleClear.bind(this))
	element.addEventListener('keyup', this.handleKeyUp.bind(this))
	element.addEventListener('keydown', this.handleKeyDown.bind(this))
	// element.addEventListener('keypress', this.handleKey)
	this.element = element;
	// this.element.camera = this;
	// window.camera_ = this;

}
FreeCamera.prototype.destroy = function() {
	// element.removeEventListener('mousemove', this.handleMouseMove)
	// element.removeEventListener('mousedown', this.handleMouseDown)
	// element.removeEventListener('mouseup', this.handleMouseUp)
	let element = this.element;
	element.removeEventListener('mouseleave', this.handleClear)
	element.removeEventListener('mouseenter', this.handleClear)
	element.removeEventListener('mouseover', this.handleClear)
	element.removeEventListener('mouseout', this.handleClear)
	element.removeEventListener('keyup', this.handleKeyUp)
	element.removeEventListener('keydown', this.handleKeyDown)
	// element.removeEventListener('keypress', this.handleKey)
}
FreeCamera.prototype.handleMouse = function(e) {
	// console.log('mouse', e, 'this', this);
	this.leftWasDown = this.leftDown;
	this.rghtWasDown = this.rghtDown;
	this.leftDown = e.buttons & 1;
	this.rghtDown = e.buttons & 2;
	if (this.lastX == undefined) {
		this.dx = 0;
		this.dy = 0;
	} else {
		this.dx = this.lastX - e.offsetX;
		this.dy = this.lastY - e.offsetY;
	}
	// console.log(this.lastX, "=>",this.dx, this.dy);
	// this.dx = e.movementX;
	// this.dy = e.movementY;
	this.lastX = e.offsetX;
	this.lastY = e.offsetY;
	// console.log(this, this.dx, this.lastX,this.keys);
}
FreeCamera.prototype.handleClear = function(e) {
	this.lastX = undefined;
	this.lastY = undefined;
	this.lastTime = undefined;
	this.leftDown = false;
	this.rghtDown = false;
	this.dx = 0;
	this.dy = 0;
	this.shift = this.alt = false;
	this.keys = [];
}
FreeCamera.prototype.handleKeyUp = function(e) {
	if (e.key == "Shift") {
		this.shift = false;
		return;
	}
	if (e.key == "Alt") {
		this.alt = false;
		return;
	}

	let idx = this.keys.indexOf(e.key);
	if (idx >= 0) {
		this.keys.splice(idx,1);
	}
}
FreeCamera.prototype.handleKeyDown = function(e) {
	// console.log(e);
	if (e.key == "Shift") {
		this.shift = true;
		return;
	}
	if (e.key == "Alt") {
		this.alt = true;
		return;
	}

	if (!e.repeat)
		this.keys.push(e.key);
}

FreeCamera.prototype.computeMatrices = function(ctx) {
	const n = .01;
	// const s = Math.tan(this.fov*.5*3.141/180);
	const s = .7;
	let u = n * s * (ctx.viewportWidth / ctx.viewportHeight), v = n * s;
	this.proj = frustum(n, 4.5, -u,u, v,-v);

	let VI = qmatrix(this.q);
	VI[3*4+0] = this.t[0];
	VI[3*4+1] = this.t[1];
	VI[3*4+2] = this.t[2];
	VI[3*4+3] = 1;
	// console.log(VI);
	// this.VI = VI;
	this.view = inverseSE3(VI);
}
FreeCamera.prototype.step = function(ctx) {
	// let {dx, dy, down} = this.updateIo();
	// this.updateIo();
	const dx = this.dx, dy = this.dy;
	const dt = 1/60.;

	if (this.leftDown) {
		const qspeed = .2 * dt;
		let dq1 = qexp([this.dy*qspeed,0,0]);
		let dq2 = qexp([0,this.dx*qspeed,0]);
		this.q = qmult(dq2, qmult(this.q, dq1));
		// this.q = qmult(this.q, qmult(dq2, dq1));
		// this.q = qmult(dq1, qmult(dq2, this.q));
	}

	let t = this.t;
	let v = this.v;

	// console.log(this.keys);
	const speed = 12.0;
	let a = [
		speed * ((this.keys.indexOf("d") != -1) ?  1 : (this.keys.indexOf("a") != -1) ? -1 : 0),
		speed * ((this.keys.indexOf("q") != -1) ? -1 : (this.keys.indexOf("e") != -1) ?  1 : 0),
		speed * ((this.keys.indexOf("w") != -1) ?  1 : (this.keys.indexOf("s") != -1) ? -1 : 0),
	];
	a = topLeft_mtv_3(this.view, a);

	if (1) {
		let drag_= 19.5 * norm(v);
		let drag = Math.min(.99/dt, drag_);
		a[0] -= drag * v[0];
		a[1] -= drag * v[1];
		a[2] -= drag * v[2];
	} else {
		let drag_ = 20.1 * dt;
		a[0] -= drag_ * v[0] * Math.abs(v[0]);
		a[1] -= drag_ * v[1] * Math.abs(v[1]);
		a[2] -= drag_ * v[2] * Math.abs(v[2]);
	}

	v[0] = v[0] + a[0]*dt;
	v[1] = v[1] + a[1]*dt;
	v[2] = v[2] + a[2]*dt;
	// console.log(v, this.keys)

	t[0] = t[0] + v[0]*dt;
	t[1] = t[1] + v[1]*dt;
	t[2] = t[2] + v[2]*dt;

	this.computeMatrices(ctx);
	this.dx = 0;
	this.dy = 0;
}
// Camera.prototype.updateIo = function() {
// }


exports.FreeCamera = FreeCamera;

// ---------------------------------------------------------------
// Quaternions
// ---------------------------------------------------------------

function qmult(p,q) {
	const a1=p[0], b1=p[1], c1=p[2], d1=p[3];
	const a2=q[0], b2=q[1], c2=q[2], d2=q[3];
	return new Float32Array([
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2]);
}
function qexp(u) {
	let n = Math.sqrt(u.map(e=>e*e).reduce((a,b)=>a+b, 0));
	if (n < 1e-16) {
		return new Float32Array([1,0,0,0]);
	}
	let k = u.slice().map(a=>a/n);
	return new Float32Array([
		Math.cos(n*.5),
		Math.sin(n*.5) * k[0],
		Math.sin(n*.5) * k[1],
		Math.sin(n*.5) * k[2]]);
}
function qlog(u) {
}
function qmatrix(u) {
	const q0=u[0], q1=u[1], q2=u[2], q3=u[3];
	return transpose(new Float32Array([
        q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3), 0,
        2*(q1*q2+q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3), 2*(q2*q3-q0*q1), 0,
        2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0*q0-q1*q1-q2*q2+q3*q3, 0,
		0,0,0,1
	]));
}

exports.qmult = qmult;
exports.qexp = qexp;
exports.qlog = qlog;
exports.qmatrix = qmatrix;

// ---------------------------------------------------------------
// Geographic conversions
// ---------------------------------------------------------------

const Earth = (() => {
	const R1         = (6378137.0);
	const R2         = (6356752.314245179);
	const a          = 1;
	const b          = R2 / R1;
	return {
		R1         : (6378137.0),
		R2         : (6356752.314245179),
		a          : 1,
		b          : R2 / R1,
		b2_over_a2 : (b*b) / (a*a),
		e2         : 1 - (b * b / a * a),
	};
})();

function uwm_to_geodetic(x) {
	return new Float32Array([
		x[0] * Math.PI,
		Math.atan(Math.exp(x[1] * Math.PI)) * 2 - Math.PI / 2,
		x[2] * Math.PI]);
}
function uwm_to_ecef(x) {
	return geodetic_to_ecef(uwm_to_geodetic(x));
}
function geodetic_to_unit_wm(x) {
	return new Float32Array([
		x[0] / Math.PI,
		Math.log(Math.tan(Math.PI/4 + x[1]*.5)) / Math.PI,
		x[2] / Math.PI
	]);
}
function ecef_to_geodetic(ecef) {
	const x = ecef[0], y = ecef[1], z = ecef[2];
	const ox = Math.atan2(y,x);
	let   k = 1. / (1. - Earth.e2);
	const p2 = x*x + y*y;
	const p = Math.sqrt(p2);
	for (let i=0; i<2; i++) {
		const c = Math.pow(((1-Earth.e2) * z*z) * (k*k) + p2, 1.5) / Earth.e2;
		k = (c + (1-Earth.e2) * z * z * Math.pow(k, 3)) / (c - p2);
	}
	const oy = Math.atan2(k*z, p);

	const rn = Earth.a / Math.sqrt(1-Earth.e2 * Math.pow(Math.sin(oy), 2));
	const sinabslat = Math.sin(Math.abs(oy));
	const coslat = Math.cos(oy);
	const oz = (Math.abs(z) + p - rn * (coslat + (1-Earth.e2) * sinabslat)) / (coslat + sinabslat);

	return new Float32Array([ ox, oy, oz ]);
}
function ecef_to_uwm(x) {
	return geodetic_to_unit_wm(ecef_to_geodetic(x));
}
function geodetic_to_ecef(g) {
	const cp = Math.cos(g[1]);
	const sp = Math.sin(g[1]);
	const cl = Math.cos(g[0]);
	const sl = Math.sin(g[0]);
	const n_phi = Earth.a / Math.sqrt(1 - Earth.e2 * sp * sp);
	return new Float32Array([
		(n_phi + g[2]) * cp * cl,
		(n_phi + g[2]) * cp * sl,
		(Earth.b2_over_a2 * n_phi + g[2]) * sp
	]);
}

exports.Earth = Earth;
exports.uwm_to_geodetic = uwm_to_geodetic;
exports.uwm_to_ecef = uwm_to_ecef;
exports.ecef_to_uwm = ecef_to_uwm;
exports.geodetic_to_ecef = geodetic_to_ecef;

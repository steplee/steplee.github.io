

export function Frustum(regl, d) {

	this.cam_imvp = d.cam_imvp;

	if (Frustum.prototype.draw_lines === undefined) {

		Frustum.prototype.draw_lines = regl({
				uniforms: {
					u_mvew: regl.context('mvew'),
					u_proj: regl.context('proj'),
					u_imvp: regl.props('cam_imvp'), // The inverse camera matrix for this frustum.
				},

				vert: `precision mediump float;
				uniform mat4 u_proj;
				uniform mat4 u_mvew;
				uniform mat4 u_imvp;
				attribute vec3 a_pos;
				void main() {
					gl_Position = u_proj * u_mvew * u_imvp * vec4(a_pos, 1.);
				}`,
				frag: `precision mediump float;
				uniform vec4 u_color;
				void main() {
					gl_FragColor = u_color;
				}`,

				attributes: {
					a_pos: { buffer: [
						-1, -1, -1,
						 1, -1, -1,
						 1,  1, -1,
						-1,  1, -1,
						-1, -1,  1,
						 1, -1,  1,
						 1,  1,  1,
						-1,  1,  1 ]},
				},

				elements: [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
				primitive: 'lines',
		});

		Frustum.prototype.draw = function() {
			this.draw_lines({cam_imvp: this.cam_imvp});
			// this.draw_image({cam_imvp: this.cam_imvp});
		}
	}

}


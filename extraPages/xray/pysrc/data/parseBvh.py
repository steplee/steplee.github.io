import numpy as np, math, sys
import json
from copy import deepcopy
_EPS4 = np.finfo(float).eps * 4.0

'''

This file exports a BVH (I'm using the mp pose prior dataset) to a numpy float32 file.
That binary file is read and used with 'posePriorDataset.py'
This script can also show the data in 3d.
This script allows augmenting each pose while exporting.

'''

# From https://github.com/matthew-brett/transforms3d/blob/bb2c0a90629e30699e55bb7fd1dd56e99d77f9e1/transforms3d/euler.py
def euler2mat(ai, aj, ak):
    i,j,k = 0,1,2

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    M[i, i] = cj*ck
    M[i, j] = sj*sc-cs
    M[i, k] = sj*cc+ss
    M[j, i] = cj*sk
    M[j, j] = sj*ss+cc
    M[j, k] = sj*cs-sc
    M[k, i] = -sj
    M[k, j] = cj*si
    M[k, k] = cj*ci
    return M

def mat2euler(mat):
    i,j,k = 0,1,2

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
    if cy > _EPS4:
        ax = math.atan2( M[k, j],  M[k, k])
        ay = math.atan2(-M[k, i],  cy)
        az = math.atan2( M[j, i],  M[i, i])
    else:
        ax = math.atan2(-M[j, k],  M[j, j])
        ay = math.atan2(-M[k, i],  cy)
        az = 0.0

    return ax, ay, az


# From https://github.com/dabeschte/npybvh/blob/master/bvh.py
import re
# from transforms3d.euler import euler2mat, mat2euler


class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.offset0 = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return 'j'+self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0
        self.iterStride = 1

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []

        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
                    joint_stack[-1].offset0[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

        # Sort, so that with the same joints we always have the same indices order,
        # no matter the order in the BVH file
        js = sorted(self.joints.items(), key=lambda t:t[0])
        self.joints = dict(js)
        # print(self.joints)
        self.jointsRev = {v.name: i for i,(k,v) in enumerate(self.joints.items())}
        # self.origJoints = deepcopy(dict(js))


    def augment(self, augmentDict):
        # js = deepcopy(self.origJoints)

        for k in augmentDict.keys(): assert k in augmentDict

        for k,mult in augmentDict.items():
            self.joints[k].offset[:] = self.joints[k].offset0[:]
        for k,mult in augmentDict.items():
            # FIXME: This or the children?
            # self.joints[k].offset[:] *= mult
            for c in self.joints[k].children:
                c.offset[:] *= mult

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)
        plt.show()

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue

            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)

        # joint_index = self.jointsValsAsList.index(joint)
        joint_index = self.jointsRev[joint.name]

        if len(joint.channels) == 0:
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        rotation = np.rad2deg(mat2euler(M))
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

        return index_offset

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))

        return p, r

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def _plot_pose(self, p, r, fig=None, ax=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-1, 59)

        plt.draw()
        plt.pause(0.00001)

    def plot_frame(self, frame, fig=None, ax=None):
        p, r = self.frame_pose(frame)
        self._plot_pose(p, r, fig, ax)

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def plot_all_frames(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.frames):
            if i % 10 == 0: print(f"frame {i}/{self.frames}")
            self.plot_frame(i, fig, ax)

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"









    def __iter__(self):
        for i in range(0,self.frames,self.iterStride):
            p,r = self.frame_pose(i)
            yield p,r

    def getInds(self):
        keys = list(self.joints.keys())
        vals = list(self.joints.values())
        inds = []
        for i,joint in enumerate(vals):
            if joint.parent:
                j = vals.index(joint.parent)
                assert j>=0
                inds.append((i,j))
        return inds



from ..render import *


def show_anim_3d(anim, augmentEach=False):
    inds = anim.getInds()

    app = SurfaceRenderer(1024,1024)
    app.init(True)
    for p,r in anim:

        if augmentEach:
            anim.augment(get_random_augment())

        verts = p.astype(np.float32) * .1

        app.startFrame()
        app.render()

        glBegin(GL_LINES)
        glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(4,0,0);
        glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,4,0);
        glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,4);
        glEnd()

        glBegin(GL_LINES)
        glColor4f(1,1,1,1)
        for a,b in inds:
            glVertex3f(verts[a,0], verts[a,1], verts[a,2])
            glVertex3f(verts[b,0], verts[b,1], verts[b,2])
        glEnd()

        app.endFrame()

def find_all_bvh_files(d):
    for root,dirs,files in os.walk(d):
        for f in files:
            if f.endswith('.bvh'): yield os.path.join(root, f)



def get_random_augment():
    def ranged(lo,hi):
        return np.random.rand() * (hi-lo) + lo
    def leftRight(k,v):
        return {'Left'+k:v, 'Right'+k: v}
    def end(k,v):
        return {k:v, k+'_end': v}
    def leftRightEnd(k,v):
        return {**end('Left'+k,v), **end('Right'+k,v)}


    r_width = (.9, 1.2)
    r_height = (.8, 1.3)

    r_legLen = (.9, 1.1)
    r_foreArmLen = (.9, 1.1)
    r_armLen = (.9, 1.1)


    Width = np.array((ranged(*r_width), 1, 1))
    # Height = np.array((1, ranged(*r_height), 1))
    Height = ranged(*r_height)
    # Height = 1

    # Multiply by width in X axis, Height in Z axis.
    hip = ranged(.9,1.1) * Width * np.array((1,Height,1))
    shoulder = ranged(.9,1.01) * Width

    arm = ranged(.9,1.1) * Height
    forearm = ranged(.9,1.1) * Height
    hand = ranged(.95,1.05) * Height

    leg = ranged(.9,1.1) * Height
    upleg = ranged(.9,1.1) * Height
    foot = ranged(.95,1.05) * Height

    head = ranged(.9,1.1)
    spine = ranged(.9,1.1) * Height


    return dict(
            # Overall width
            **leftRight('Shoulder', shoulder),
            Hips=hip,

            # Arms
            **leftRight('Arm', arm),
            **leftRight('ForeArm', forearm),
            **leftRightEnd('Hand', forearm),
            **leftRightEnd('HandThumb1', forearm),

            # Legs
            **leftRight('UpLeg', upleg),
            **leftRight('Leg', leg),
            **leftRight('Foot', foot),
            **leftRightEnd('ToeBase', foot),

            # Head
            Head=head*Height,
            Head_end=head,
            Spine=spine,

            )

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--inputDir', default='')
    parser.add_argument('--exportDir', default='')
    parser.add_argument('--exportStride', default=2, type=int)

    # Show a 3d animation of the one file
    parser.add_argument('--singleInput', default='')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    try: os.makedirs(args.exportDir)
    except: pass

    if args.show:
        anim = Bvh()
        if args.singleInput:
            anim.parse_file(args.singleInput)

            # augd = get_random_augment()
            # anim.augment(get_random_augment())
            # anim.augment({ 'LeftShoulder': 2, 'RightShoulder': 2, })
            show_anim_3d(anim, augmentEach=args.augment)


    if args.exportDir:
        data = []
        files = list(find_all_bvh_files(args.inputDir))

        try:
            for fi,f in enumerate(files):
                print(f' on {fi}/{len(files)}')
                anim = Bvh()
                anim.parse_file(f)
                anim.iterStride = args.exportStride
                for i,(p,r) in enumerate(anim):
                    if args.augment: anim.augment(get_random_augment())
                    p = p.astype(np.float32)
                    data.append(p)
                print(anim.joint_names())
                # print(anim.getInds())
        except KeyboardInterrupt:
            print(' - Interrupted... still saving partial results.')
            pass

        data = np.stack(data)

        out_file = os.path.join(args.exportDir,'data')
        print(f' - Saving data of shape {data.shape} to {out_file}.npz')
        meta = dict(srcFiles=files, stride=args.exportStride)
        inds = np.array(anim.getInds()).astype(np.uint16).reshape(-1)
        np.savez(out_file, data=data, inds=inds, joints=list(anim.joints.keys()), meta=meta)

        '''
        import json
        meta = dict(srcFiles=files, stride=args.exportStride, inds=anim.getInds())
        out_file = os.path.join(args.exportDir,'meta.json')
        print(f' - Saving meta {out_file}')
        with open(out_file, 'w') as fp:
            json.dump(meta,fp)
        '''

    sys.exit(0)




if __name__ == '__main__':
    # create Bvh parser
    anim = Bvh()
    # parser file
    import sys
    anim.parse_file(sys.argv[-1])

    # draw the skeleton in T-pose
    # anim.plot_hierarchy()
    print(anim.joints)
    print('inds', anim.getInds())

    '''
    # extract single frame pose: axis0=joint, axis1=positionXYZ/rotationXYZ
    p, r = anim.frame_pose(0)

    # extract all poses: axis0=frame, axis1=joint, axis2=positionXYZ/rotationXYZ
    all_p, all_r = anim.all_frame_poses()

    # print all joints, their positions and orientations
    for _p, _r, _j in zip(p, r, anim.joint_names()):
        print(f"{_j}: p={_p}, r={_r}")

    # draw the skeleton for the given frame
    # anim.plot_frame(22)

    # show full animation
    anim.plot_all_frames()
    '''
    show_anim_3d(anim)


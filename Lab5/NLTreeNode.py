import numpy as np

out = 0


class NLTreeNode:
    # ========================== constructor ==============================

    def __init__(self, type, data):
        self.data = data
        self.type = type
        self.nchd = 0
        self.children = []
        self.nprt = 0
        self.parents = []

    # def __init__(self, data, nchd, children, nprt, parents):
    #  self.data = data
    #  self.nchd = nchd
    #  self.children = children
    #  self.nprt = nprt
    #  self.parents = parents

    def add_chd(self, chd):
        self.nchd = self.nchd + 1
        self.children.append(chd)
        chd.add_prt(self)

    def add_prt(self, prt):
        self.nprt = self.nprt + 1
        self.parents.append(prt)

    def set_data(self, data):
        self.data = data

    def set_type(self, type):
        self.type = type

    # - - - - - - methods to evaluate the (tree of ) nodes - - - - -

    def eval_val(self, x):
        """
        this traverses the tree and evaluates the function
        """
        if self.type == "n":
            # print(f"n node: {self.data:f}")
            return self.data
        elif self.type == "v":
            ix = int(self.data)
            # print(f"v node: ix= {ix:d}, {x[ix]:f}")
            return x[ix]
        elif self.type == "+":
            f1 = self.children[0].eval_val(x)
            f2 = self.children[1].eval_val(x)
            # print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
            return f1+f2
        elif self.type == "-":
            f1 = self.children[0].eval_val(x)
            f2 = self.children[1].eval_val(x)
            return f1-f2
        elif self.type == "un-":
            f1 = self.children[0].eval_val(x)
            return -f1
        elif self.type == "*":
            f1 = self.children[0].eval_val(x)
            f2 = self.children[1].eval_val(x)
            return f1*f2
        elif self.type == "/":
            f1 = self.children[0].eval_val(x)
            f2 = self.children[1].eval_val(x)
            return f1/f2
        elif self.type == "^":
            # print("^: eval first child: ")
            f1 = self.children[0].eval_val(x)
            # print(f"f1 = {f1:f}")
            # print("^: eval second child: ")
            f2 = self.children[1].eval_val(x)
            # print(f"f2 = {f2:f}")
            # print(f"^ node: {f1:f}^{f2:f} = {f1**f2:f}")

            return f1**f2
        elif self.type == "exp":
            f1 = self.children[0].eval_val(x)
            return np.exp(f1)
        elif self.type == "sin":
            f1 = self.children[0].eval_val(x)
            return np.sin(f1)
        elif self.type == "sum":
            ret = 0
            for i in range(self.nchd):
                ret += self.children[i].eval_val(x)
            return ret
        else:
            raise ValueError(f"eval_val(): Operation not implemented yet: \n{self.type:s}")

    def eval_grad_forwardADTree(self, x):
        """
        this traverses the tree and evaluates the gradient
        """

        n = x.size  # get dimension of x
        if self.type == "n":
            # print(f"n node: {self.data:f}")
            gd = np.zeros(n)
            return (self.data, gd)
        elif self.type == "v":
            ix = int(self.data)
            # print(f"v node: ix= {ix:d}, {x[ix]:f}")
            gd = np.zeros(n)
            gd[ix] = 1
            # print (x[ix], gd)
            return (x[ix], gd)
        elif self.type == "+":
            (f1, g1) = self.children[0].eval_grad_forwardADTree(x)
            (f2, g2) = self.children[1].eval_grad_forwardADTree(x)
            # print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
            # print(f1+f2, g1+g2)
            return (f1+f2, g1+g2)
        elif self.type == "-":
            (f1, g1) = self.children[0].eval_grad_forwardADTree(x)
            (f2, g2) = self.children[1].eval_grad_forwardADTree(x)
            # print(f"- node: {f1:f}+{f2:f} = {f1+f2:f}")
            # print(f1-f2,g1-g2)
            return (f1-f2, g1-g2)
        elif self.type == "un-":
            (f1, g1) = self.children[0].eval_grad_forwardADTree(x)
            return (-f1, -g1)
        elif self.type == "*":
            (f1, g1) = self.children[0].eval_grad_forwardADTree(x)
            (f2, g2) = self.children[1].eval_grad_forwardADTree(x)
            # print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
            # print("* node:")
            # print(f1*f2, f2*g1+f1*g2)
            return (f1*f2, f2*g1+f1*g2)

        elif self.type == "^":
            if out >= 1:
                print("Eval grad ^ node")
            (f1, g1) = self.children[0].eval_grad_forwardADTree(x)
            (f2, g2) = self.children[1].eval_grad_forwardADTree(x)
            # print(f"f1 = {f1:f}, g1 = {g1}")
            # print(f"f2 = {f2:f}, g2 = {g2}")
            # print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
            val = f1**f2
            gd = f2*(f1**(f2-1))*g1
            if np.linalg.norm(g2-np.zeros(n)) > 1e-6:
                gd = gd + (f1**f2)*np.log(f1)*g2
            # print(val, gd)
            return (val, gd)
        elif self.type == "sum":
            val = 0
            gd = np.zeros(n)
            for i in range(self.nchd):
                vi, gdi = self.children[i].eval_grad_forwardADTree(x)
                val += vi
                gd += gdi
            return (val, gd)
        else:
            raise ValueError(f"eval_grad(): Operation not implemented yet: \n{self.type:s}")

    def eval_hess_forwardADTree(self, x):
        """
        this traverses the tree and evaluates the Hessian
        """
        n = x.size  # get dimension of x
        if self.type == "n":
            # print(f"n node: {self.data:f}")
            gd = np.zeros(n)
            H = np.zeros((n, n))
            return (self.data, gd, H)
        elif self.type == "v":
            ix = int(self.data)
            if out >= 1:
                print(f"v node: ix= {ix:d}, {x[ix]:f}")
            gd = np.zeros(n)
            gd[ix] = 1
            if out >= 1:
                print(x[ix], gd)
            H = np.zeros((n, n))
            return (x[ix], gd, H)
        elif self.type == "+":
            (f1, g1, H1) = self.children[0].eval_hess_forwardADTree(x)
            (f2, g2, H2) = self.children[1].eval_hess_forwardADTree(x)
            if out >= 1:
                print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
                print(f1+f2, g1+g2, H1+H2)
            return (f1+f2, g1+g2, H1+H2)
        elif self.type == "-":
            (f1, g1, H1) = self.children[0].eval_hess_forwardADTree(x)
            (f2, g2, H2) = self.children[1].eval_hess_forwardADTree(x)
            if out >= 1:
                print(f"- node: {f1:f}+{f2:f} = {f1+f2:f}")
                print(f1-f2, g1-g2, H1-H2)
            return (f1-f2, g1-g2, H1-H2)
        elif self.type == "un-":
            (f1, g1, H1) = self.children[0].eval_hess_forwardADTree(x)
            return (-f1, -g1, -H1)
        elif self.type == "*":
            (f1, g1, H1) = self.children[0].eval_hess_forwardADTree(x)
            (f2, g2, H2) = self.children[1].eval_hess_forwardADTree(x)
            # print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
            if out >= 1:
                print("* node:")
                print(f1*f2, f2*g1+f1*g2)
            Hr = f1*H2 + f2*H1 + np.outer(g1, g2) + np.outer(g2, g1)
            return (f1*f2, f2*g1+f1*g2, Hr)
        elif self.type == "^":
            if out >= 1:
                print("Eval grad ^ node")
            (f1, g1, H1) = self.children[0].eval_hess_forwardADTree(x)
            (f2, g2, H2) = self.children[1].eval_hess_forwardADTree(x)
            if out >= 1:
                print(f"f1 = {f1:f}, g1 = {g1}")
                print(f"f2 = {f2:f}, g2 = {g2}")
            print(f"+ node: {f1:f}+{f2:f} = {f1+f2:f}")
            val = f1**f2
            gd = f2*(f1**(f2-1))*g1
            if np.linalg.norm(g2-np.zeros(n)) > 1e-6:
                gd = gd + (f1**f2)*np.log(f1)*g2
            print(val, gd)
            Hr = f2*(f1**(f2-2))*(f1*H1 + (f2-1)*np.outer(g1, g1))
            if np.linalg.norm(g2-np.zeros(n)) > 1e-6:
                print("Operation not implemented yet: ", end="")
                print("Hessian for ^ with non-constant exponent")
            return (val, gd, Hr)
        elif self.type == "sum":
            val = 0
            gd = np.zeros(n)
            H = np.zeros((n, n))
            for i in range(self.nchd):
                vi, gdi, Hi = self.children[i].eval_hess_forwardADTree(x)
                val += vi
                gd += gdi
                H += Hi
            return (val, gd, H)
        else:
            raise ValueError(f"eval_hess(): Operation not implemented yet: \n{self.type:s}")

    # - - - - - - - -  print methods - - - - - - - -

    def to_string(self):
        # for brackets need to check if node above has higher precedence
        # -> then include brackets
        s = []
        for i in range(self.nchd):
            s.append(self.children[i].to_string())

        if self.type == "n":
            return str(self.data)
        elif self.type == "v":
            return "x["+str(self.data)+"]"
        elif self.type == "+":
            return "("+s[0]+"+"+s[1]+")"
        elif self.type == "-":
            return "("+s[0]+"-"+s[1]+")"
        elif self.type == "un-":
            return "-"+s[0]
        elif self.type == "*":
            return s[0]+"*"+s[1]
        elif self.type == "/":
            return s[0]+"/"+s[1]
        elif self.type == "^":
            return s[0]+"^"+s[1]
        elif self.type == "exp":
            return "exp("+s[0]+")"
        elif self.type == "sin":
            return "sin("+s[0]+")"
        elif self.type == "sum":
            ret = "("
            for i in range(self.nchd):
                if i > 0:
                    ret += "+"
                ret += s[i]
            ret += ")"
            return ret
        else:
            raise ValueError(f"to_string(): Operation not implemented yet: {self.type:s}")
        print(self.type)

    def get_flat_tree(self, nvar):
        """
        This method gets a flat representation of the tree (as an array of
        nodes) and also removes duplicate variable nodes
        """
        flat = []
        varnodes = [None]*nvar
        self._add_node_to_flat(flat, varnodes)

        cnt = 0
        for nd in flat:
            if out >= 1:
                print(f"nd[{cnt:d}] = {nd.type:s} ({nd.data:f}), chd = {nd.nchd:d}, nprt = {nd.nprt:d}")
            cnt += 1
        return flat

    def _add_node_to_flat(self, flat, varnodes):
        """
        recursive helper method to create flat tree
        """
        if self.type == "v":
            ix = int(self.data)
            if out >= 1:
                print(f"Found variable {ix:d}")
            if varnodes[ix] is None:
                if out >= 1:
                    print(f"Not registered yet: now at {len(flat):d}")
                varnodes[ix] = len(flat)
                flat.append(self)
            else:
                # this is the important bit: var node has already been
                # registered: replace the node on the tree by the already
                # registered one (and add a new parent). Also need to replace
                # the node on the flat list
                if out >= 1:
                    print(f"var node {ix:d} already on flat tree at position {varnodes[ix]:d}")
                pos = varnodes[ix]
                if out >= 1:
                    print(f"flat[{pos:d}] is type = {flat[pos].type:s}, ix = {int(flat[pos].data):d}")
                prnt_nd = self.parents[0]
                # how do we know which child this is?
        else:
            flat.append(self)
        # print(f"Called _add_node_to_flat. type = {self.type:s}, nchd = {self.nchd:d}")

        for i in range(self.nchd):
            chd = self.children[i]
            chd._add_node_to_flat(flat, varnodes)

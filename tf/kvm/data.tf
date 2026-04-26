data "openstack_networking_network_v2" "sharednet1" {
  name = "sharednet1"
}

data "openstack_networking_subnet_v2" "sharednet1_subnet" {
  name = "sharednet1-subnet"
}

data "openstack_networking_secgroup_v2" "allow_ssh" {
  name = "allow-ssh"
}

data "openstack_networking_secgroup_v2" "allow_30900" {
  name = "allow-30900"
}

data "openstack_networking_secgroup_v2" "allow_30808" {
  name = "allow-30808"
}

data "openstack_networking_secgroup_v2" "allow_30500" {
  name = "allow-30500"
}

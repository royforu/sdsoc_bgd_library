// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2019.1
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="a2_s2mm_data_copy,hls_ip_2019_1,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=1,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xc7z045-ffg900-2,HLS_INPUT_CLOCK=10.000000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=7.300000,HLS_SYN_LAT=-1,HLS_SYN_TPT=none,HLS_SYN_MEM=2,HLS_SYN_DSP=0,HLS_SYN_FF=654,HLS_SYN_LUT=675,HLS_VERSION=2019_1}" *)

module a2_s2mm_data_copy (
        ap_clk,
        ap_rst_n,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_buf_r_AWVALID,
        m_axi_buf_r_AWREADY,
        m_axi_buf_r_AWADDR,
        m_axi_buf_r_AWID,
        m_axi_buf_r_AWLEN,
        m_axi_buf_r_AWSIZE,
        m_axi_buf_r_AWBURST,
        m_axi_buf_r_AWLOCK,
        m_axi_buf_r_AWCACHE,
        m_axi_buf_r_AWPROT,
        m_axi_buf_r_AWQOS,
        m_axi_buf_r_AWREGION,
        m_axi_buf_r_AWUSER,
        m_axi_buf_r_WVALID,
        m_axi_buf_r_WREADY,
        m_axi_buf_r_WDATA,
        m_axi_buf_r_WSTRB,
        m_axi_buf_r_WLAST,
        m_axi_buf_r_WID,
        m_axi_buf_r_WUSER,
        m_axi_buf_r_ARVALID,
        m_axi_buf_r_ARREADY,
        m_axi_buf_r_ARADDR,
        m_axi_buf_r_ARID,
        m_axi_buf_r_ARLEN,
        m_axi_buf_r_ARSIZE,
        m_axi_buf_r_ARBURST,
        m_axi_buf_r_ARLOCK,
        m_axi_buf_r_ARCACHE,
        m_axi_buf_r_ARPROT,
        m_axi_buf_r_ARQOS,
        m_axi_buf_r_ARREGION,
        m_axi_buf_r_ARUSER,
        m_axi_buf_r_RVALID,
        m_axi_buf_r_RREADY,
        m_axi_buf_r_RDATA,
        m_axi_buf_r_RLAST,
        m_axi_buf_r_RID,
        m_axi_buf_r_RUSER,
        m_axi_buf_r_RRESP,
        m_axi_buf_r_BVALID,
        m_axi_buf_r_BREADY,
        m_axi_buf_r_BRESP,
        m_axi_buf_r_BID,
        m_axi_buf_r_BUSER,
        fifo_TDATA,
        fifo_TVALID,
        fifo_TREADY,
        len,
        buf_offset
);

parameter    ap_ST_fsm_state1 = 7'd1;
parameter    ap_ST_fsm_pp0_stage0 = 7'd2;
parameter    ap_ST_fsm_state4 = 7'd4;
parameter    ap_ST_fsm_state5 = 7'd8;
parameter    ap_ST_fsm_state6 = 7'd16;
parameter    ap_ST_fsm_state7 = 7'd32;
parameter    ap_ST_fsm_state8 = 7'd64;
parameter    C_M_AXI_BUF_R_ID_WIDTH = 1;
parameter    C_M_AXI_BUF_R_ADDR_WIDTH = 32;
parameter    C_M_AXI_BUF_R_DATA_WIDTH = 32;
parameter    C_M_AXI_BUF_R_AWUSER_WIDTH = 1;
parameter    C_M_AXI_BUF_R_ARUSER_WIDTH = 1;
parameter    C_M_AXI_BUF_R_WUSER_WIDTH = 1;
parameter    C_M_AXI_BUF_R_RUSER_WIDTH = 1;
parameter    C_M_AXI_BUF_R_BUSER_WIDTH = 1;
parameter    C_M_AXI_BUF_R_USER_VALUE = 0;
parameter    C_M_AXI_BUF_R_PROT_VALUE = 0;
parameter    C_M_AXI_BUF_R_CACHE_VALUE = 3;
parameter    C_M_AXI_DATA_WIDTH = 32;

parameter C_M_AXI_BUF_R_WSTRB_WIDTH = (32 / 8);
parameter C_M_AXI_WSTRB_WIDTH = (32 / 8);

input   ap_clk;
input   ap_rst_n;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_buf_r_AWVALID;
input   m_axi_buf_r_AWREADY;
output  [C_M_AXI_BUF_R_ADDR_WIDTH - 1:0] m_axi_buf_r_AWADDR;
output  [C_M_AXI_BUF_R_ID_WIDTH - 1:0] m_axi_buf_r_AWID;
output  [7:0] m_axi_buf_r_AWLEN;
output  [2:0] m_axi_buf_r_AWSIZE;
output  [1:0] m_axi_buf_r_AWBURST;
output  [1:0] m_axi_buf_r_AWLOCK;
output  [3:0] m_axi_buf_r_AWCACHE;
output  [2:0] m_axi_buf_r_AWPROT;
output  [3:0] m_axi_buf_r_AWQOS;
output  [3:0] m_axi_buf_r_AWREGION;
output  [C_M_AXI_BUF_R_AWUSER_WIDTH - 1:0] m_axi_buf_r_AWUSER;
output   m_axi_buf_r_WVALID;
input   m_axi_buf_r_WREADY;
output  [C_M_AXI_BUF_R_DATA_WIDTH - 1:0] m_axi_buf_r_WDATA;
output  [C_M_AXI_BUF_R_WSTRB_WIDTH - 1:0] m_axi_buf_r_WSTRB;
output   m_axi_buf_r_WLAST;
output  [C_M_AXI_BUF_R_ID_WIDTH - 1:0] m_axi_buf_r_WID;
output  [C_M_AXI_BUF_R_WUSER_WIDTH - 1:0] m_axi_buf_r_WUSER;
output   m_axi_buf_r_ARVALID;
input   m_axi_buf_r_ARREADY;
output  [C_M_AXI_BUF_R_ADDR_WIDTH - 1:0] m_axi_buf_r_ARADDR;
output  [C_M_AXI_BUF_R_ID_WIDTH - 1:0] m_axi_buf_r_ARID;
output  [7:0] m_axi_buf_r_ARLEN;
output  [2:0] m_axi_buf_r_ARSIZE;
output  [1:0] m_axi_buf_r_ARBURST;
output  [1:0] m_axi_buf_r_ARLOCK;
output  [3:0] m_axi_buf_r_ARCACHE;
output  [2:0] m_axi_buf_r_ARPROT;
output  [3:0] m_axi_buf_r_ARQOS;
output  [3:0] m_axi_buf_r_ARREGION;
output  [C_M_AXI_BUF_R_ARUSER_WIDTH - 1:0] m_axi_buf_r_ARUSER;
input   m_axi_buf_r_RVALID;
output   m_axi_buf_r_RREADY;
input  [C_M_AXI_BUF_R_DATA_WIDTH - 1:0] m_axi_buf_r_RDATA;
input   m_axi_buf_r_RLAST;
input  [C_M_AXI_BUF_R_ID_WIDTH - 1:0] m_axi_buf_r_RID;
input  [C_M_AXI_BUF_R_RUSER_WIDTH - 1:0] m_axi_buf_r_RUSER;
input  [1:0] m_axi_buf_r_RRESP;
input   m_axi_buf_r_BVALID;
output   m_axi_buf_r_BREADY;
input  [1:0] m_axi_buf_r_BRESP;
input  [C_M_AXI_BUF_R_ID_WIDTH - 1:0] m_axi_buf_r_BID;
input  [C_M_AXI_BUF_R_BUSER_WIDTH - 1:0] m_axi_buf_r_BUSER;
input  [31:0] fifo_TDATA;
input   fifo_TVALID;
output   fifo_TREADY;
input  [31:0] len;
input  [31:0] buf_offset;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
(* fsm_encoding = "none" *) reg   [6:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [31:0] fifo_0_data_out;
wire    fifo_0_vld_in;
wire    fifo_0_vld_out;
wire    fifo_0_ack_in;
reg    fifo_0_ack_out;
reg   [31:0] fifo_0_payload_A;
reg   [31:0] fifo_0_payload_B;
reg    fifo_0_sel_rd;
reg    fifo_0_sel_wr;
wire    fifo_0_sel;
wire    fifo_0_load_A;
wire    fifo_0_load_B;
reg   [1:0] fifo_0_state;
wire    fifo_0_state_cmp_full;
reg    buf_r_blk_n_AW;
reg    buf_r_blk_n_W;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
wire    ap_block_pp0_stage0;
reg   [0:0] icmp_ln42_reg_168;
reg    buf_r_blk_n_B;
wire    ap_CS_fsm_state8;
reg   [0:0] icmp_ln44_reg_182;
reg    fifo_TDATA_blk_n;
reg    ap_enable_reg_pp0_iter0;
wire   [0:0] icmp_ln42_fu_140_p2;
reg    buf_r_AWVALID;
wire    buf_r_AWREADY;
reg    buf_r_WVALID;
wire    buf_r_WREADY;
wire    buf_r_ARREADY;
wire    buf_r_RVALID;
wire   [31:0] buf_r_RDATA;
wire    buf_r_RLAST;
wire   [0:0] buf_r_RID;
wire   [0:0] buf_r_RUSER;
wire   [1:0] buf_r_RRESP;
wire    buf_r_BVALID;
reg    buf_r_BREADY;
wire   [1:0] buf_r_BRESP;
wire   [0:0] buf_r_BID;
wire   [0:0] buf_r_BUSER;
reg   [30:0] i_0_reg_104;
reg    ap_block_state2_pp0_stage0_iter0;
wire    ap_block_state3_pp0_stage0_iter1;
reg    ap_block_state3_io;
reg    ap_block_pp0_stage0_11001;
wire   [30:0] i_fu_145_p2;
reg   [31:0] fifo_read_reg_177;
wire   [0:0] icmp_ln44_fu_151_p2;
wire    ap_CS_fsm_state4;
reg    ap_block_pp0_stage0_subdone;
reg    ap_condition_pp0_exit_iter0_state2;
wire   [31:0] empty_fu_125_p1;
reg    ap_block_state8;
reg    ap_block_pp0_stage0_01001;
wire   [29:0] buf_offset1_fu_115_p4;
wire   [31:0] zext_ln42_fu_136_p1;
reg   [6:0] ap_NS_fsm;
reg    ap_idle_pp0;
wire    ap_enable_pp0;

// power-on initialization
initial begin
#0 ap_CS_fsm = 7'd1;
#0 fifo_0_sel_rd = 1'b0;
#0 fifo_0_sel_wr = 1'b0;
#0 fifo_0_state = 2'd0;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter0 = 1'b0;
end

a2_s2mm_data_copy_buf_r_m_axi #(
    .CONSERVATIVE( 0 ),
    .USER_DW( 32 ),
    .USER_AW( 32 ),
    .USER_MAXREQS( 5 ),
    .NUM_READ_OUTSTANDING( 16 ),
    .NUM_WRITE_OUTSTANDING( 16 ),
    .MAX_READ_BURST_LENGTH( 16 ),
    .MAX_WRITE_BURST_LENGTH( 16 ),
    .C_M_AXI_ID_WIDTH( C_M_AXI_BUF_R_ID_WIDTH ),
    .C_M_AXI_ADDR_WIDTH( C_M_AXI_BUF_R_ADDR_WIDTH ),
    .C_M_AXI_DATA_WIDTH( C_M_AXI_BUF_R_DATA_WIDTH ),
    .C_M_AXI_AWUSER_WIDTH( C_M_AXI_BUF_R_AWUSER_WIDTH ),
    .C_M_AXI_ARUSER_WIDTH( C_M_AXI_BUF_R_ARUSER_WIDTH ),
    .C_M_AXI_WUSER_WIDTH( C_M_AXI_BUF_R_WUSER_WIDTH ),
    .C_M_AXI_RUSER_WIDTH( C_M_AXI_BUF_R_RUSER_WIDTH ),
    .C_M_AXI_BUSER_WIDTH( C_M_AXI_BUF_R_BUSER_WIDTH ),
    .C_USER_VALUE( C_M_AXI_BUF_R_USER_VALUE ),
    .C_PROT_VALUE( C_M_AXI_BUF_R_PROT_VALUE ),
    .C_CACHE_VALUE( C_M_AXI_BUF_R_CACHE_VALUE ))
s2mm_data_copy_buf_r_m_axi_U(
    .AWVALID(m_axi_buf_r_AWVALID),
    .AWREADY(m_axi_buf_r_AWREADY),
    .AWADDR(m_axi_buf_r_AWADDR),
    .AWID(m_axi_buf_r_AWID),
    .AWLEN(m_axi_buf_r_AWLEN),
    .AWSIZE(m_axi_buf_r_AWSIZE),
    .AWBURST(m_axi_buf_r_AWBURST),
    .AWLOCK(m_axi_buf_r_AWLOCK),
    .AWCACHE(m_axi_buf_r_AWCACHE),
    .AWPROT(m_axi_buf_r_AWPROT),
    .AWQOS(m_axi_buf_r_AWQOS),
    .AWREGION(m_axi_buf_r_AWREGION),
    .AWUSER(m_axi_buf_r_AWUSER),
    .WVALID(m_axi_buf_r_WVALID),
    .WREADY(m_axi_buf_r_WREADY),
    .WDATA(m_axi_buf_r_WDATA),
    .WSTRB(m_axi_buf_r_WSTRB),
    .WLAST(m_axi_buf_r_WLAST),
    .WID(m_axi_buf_r_WID),
    .WUSER(m_axi_buf_r_WUSER),
    .ARVALID(m_axi_buf_r_ARVALID),
    .ARREADY(m_axi_buf_r_ARREADY),
    .ARADDR(m_axi_buf_r_ARADDR),
    .ARID(m_axi_buf_r_ARID),
    .ARLEN(m_axi_buf_r_ARLEN),
    .ARSIZE(m_axi_buf_r_ARSIZE),
    .ARBURST(m_axi_buf_r_ARBURST),
    .ARLOCK(m_axi_buf_r_ARLOCK),
    .ARCACHE(m_axi_buf_r_ARCACHE),
    .ARPROT(m_axi_buf_r_ARPROT),
    .ARQOS(m_axi_buf_r_ARQOS),
    .ARREGION(m_axi_buf_r_ARREGION),
    .ARUSER(m_axi_buf_r_ARUSER),
    .RVALID(m_axi_buf_r_RVALID),
    .RREADY(m_axi_buf_r_RREADY),
    .RDATA(m_axi_buf_r_RDATA),
    .RLAST(m_axi_buf_r_RLAST),
    .RID(m_axi_buf_r_RID),
    .RUSER(m_axi_buf_r_RUSER),
    .RRESP(m_axi_buf_r_RRESP),
    .BVALID(m_axi_buf_r_BVALID),
    .BREADY(m_axi_buf_r_BREADY),
    .BRESP(m_axi_buf_r_BRESP),
    .BID(m_axi_buf_r_BID),
    .BUSER(m_axi_buf_r_BUSER),
    .ACLK(ap_clk),
    .ARESET(ap_rst_n_inv),
    .ACLK_EN(1'b1),
    .I_ARVALID(1'b0),
    .I_ARREADY(buf_r_ARREADY),
    .I_ARADDR(32'd0),
    .I_ARID(1'd0),
    .I_ARLEN(32'd0),
    .I_ARSIZE(3'd0),
    .I_ARLOCK(2'd0),
    .I_ARCACHE(4'd0),
    .I_ARQOS(4'd0),
    .I_ARPROT(3'd0),
    .I_ARUSER(1'd0),
    .I_ARBURST(2'd0),
    .I_ARREGION(4'd0),
    .I_RVALID(buf_r_RVALID),
    .I_RREADY(1'b0),
    .I_RDATA(buf_r_RDATA),
    .I_RID(buf_r_RID),
    .I_RUSER(buf_r_RUSER),
    .I_RRESP(buf_r_RRESP),
    .I_RLAST(buf_r_RLAST),
    .I_AWVALID(buf_r_AWVALID),
    .I_AWREADY(buf_r_AWREADY),
    .I_AWADDR(empty_fu_125_p1),
    .I_AWID(1'd0),
    .I_AWLEN(len),
    .I_AWSIZE(3'd0),
    .I_AWLOCK(2'd0),
    .I_AWCACHE(4'd0),
    .I_AWQOS(4'd0),
    .I_AWPROT(3'd0),
    .I_AWUSER(1'd0),
    .I_AWBURST(2'd0),
    .I_AWREGION(4'd0),
    .I_WVALID(buf_r_WVALID),
    .I_WREADY(buf_r_WREADY),
    .I_WDATA(fifo_read_reg_177),
    .I_WID(1'd0),
    .I_WUSER(1'd0),
    .I_WLAST(1'b0),
    .I_WSTRB(4'd15),
    .I_BVALID(buf_r_BVALID),
    .I_BREADY(buf_r_BREADY),
    .I_BRESP(buf_r_BRESP),
    .I_BID(buf_r_BID),
    .I_BUSER(buf_r_BUSER)
);

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_enable_reg_pp0_iter0 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_condition_pp0_exit_iter0_state2) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter0 <= 1'b0;
        end else if ((~((buf_r_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter0 <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_condition_pp0_exit_iter0_state2))) begin
            ap_enable_reg_pp0_iter1 <= (1'b1 ^ ap_condition_pp0_exit_iter0_state2);
        end else if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end else if ((~((buf_r_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        fifo_0_sel_rd <= 1'b0;
    end else begin
        if (((fifo_0_ack_out == 1'b1) & (fifo_0_vld_out == 1'b1))) begin
            fifo_0_sel_rd <= ~fifo_0_sel_rd;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        fifo_0_sel_wr <= 1'b0;
    end else begin
        if (((fifo_0_ack_in == 1'b1) & (fifo_0_vld_in == 1'b1))) begin
            fifo_0_sel_wr <= ~fifo_0_sel_wr;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        fifo_0_state <= 2'd0;
    end else begin
        if ((((fifo_0_state == 2'd2) & (fifo_0_vld_in == 1'b0)) | ((fifo_0_state == 2'd3) & (fifo_0_vld_in == 1'b0) & (fifo_0_ack_out == 1'b1)))) begin
            fifo_0_state <= 2'd2;
        end else if ((((fifo_0_state == 2'd1) & (fifo_0_ack_out == 1'b0)) | ((fifo_0_state == 2'd3) & (fifo_0_ack_out == 1'b0) & (fifo_0_vld_in == 1'b1)))) begin
            fifo_0_state <= 2'd1;
        end else if (((~((fifo_0_vld_in == 1'b0) & (fifo_0_ack_out == 1'b1)) & ~((fifo_0_ack_out == 1'b0) & (fifo_0_vld_in == 1'b1)) & (fifo_0_state == 2'd3)) | ((fifo_0_state == 2'd1) & (fifo_0_ack_out == 1'b1)) | ((fifo_0_state == 2'd2) & (fifo_0_vld_in == 1'b1)))) begin
            fifo_0_state <= 2'd3;
        end else begin
            fifo_0_state <= 2'd2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln42_fu_140_p2 == 1'd1))) begin
        i_0_reg_104 <= i_fu_145_p2;
    end else if ((~((buf_r_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        i_0_reg_104 <= 31'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((fifo_0_load_A == 1'b1)) begin
        fifo_0_payload_A <= fifo_TDATA;
    end
end

always @ (posedge ap_clk) begin
    if ((fifo_0_load_B == 1'b1)) begin
        fifo_0_payload_B <= fifo_TDATA;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln42_fu_140_p2 == 1'd1))) begin
        fifo_read_reg_177 <= fifo_0_data_out;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        icmp_ln42_reg_168 <= icmp_ln42_fu_140_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        icmp_ln44_reg_182 <= icmp_ln44_fu_151_p2;
    end
end

always @ (*) begin
    if ((icmp_ln42_fu_140_p2 == 1'd0)) begin
        ap_condition_pp0_exit_iter0_state2 = 1'b1;
    end else begin
        ap_condition_pp0_exit_iter0_state2 = 1'b0;
    end
end

always @ (*) begin
    if ((~((buf_r_BVALID == 1'b0) & (icmp_ln44_reg_182 == 1'd0)) & (1'b1 == ap_CS_fsm_state8))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((buf_r_BVALID == 1'b0) & (icmp_ln44_reg_182 == 1'd0)) & (1'b1 == ap_CS_fsm_state8))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((~((buf_r_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        buf_r_AWVALID = 1'b1;
    end else begin
        buf_r_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if ((~((buf_r_BVALID == 1'b0) & (icmp_ln44_reg_182 == 1'd0)) & (1'b1 == ap_CS_fsm_state8) & (icmp_ln44_reg_182 == 1'd0))) begin
        buf_r_BREADY = 1'b1;
    end else begin
        buf_r_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln42_reg_168 == 1'd1))) begin
        buf_r_WVALID = 1'b1;
    end else begin
        buf_r_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        buf_r_blk_n_AW = m_axi_buf_r_AWREADY;
    end else begin
        buf_r_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state8) & (icmp_ln44_reg_182 == 1'd0))) begin
        buf_r_blk_n_B = m_axi_buf_r_BVALID;
    end else begin
        buf_r_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0) & (icmp_ln42_reg_168 == 1'd1))) begin
        buf_r_blk_n_W = m_axi_buf_r_WREADY;
    end else begin
        buf_r_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln42_fu_140_p2 == 1'd1))) begin
        fifo_0_ack_out = 1'b1;
    end else begin
        fifo_0_ack_out = 1'b0;
    end
end

always @ (*) begin
    if ((fifo_0_sel == 1'b1)) begin
        fifo_0_data_out = fifo_0_payload_B;
    end else begin
        fifo_0_data_out = fifo_0_payload_A;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0) & (icmp_ln42_fu_140_p2 == 1'd1))) begin
        fifo_TDATA_blk_n = fifo_0_state[1'd0];
    end else begin
        fifo_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((buf_r_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_pp0_stage0 : begin
            if (~((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (icmp_ln42_fu_140_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (icmp_ln42_fu_140_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((1'b1 == ap_CS_fsm_state4) & (icmp_ln44_fu_151_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            if ((~((buf_r_BVALID == 1'b0) & (icmp_ln44_reg_182 == 1'd0)) & (1'b1 == ap_CS_fsm_state8))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd6];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((fifo_0_vld_out == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (icmp_ln42_fu_140_p2 == 1'd1));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = (((fifo_0_vld_out == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (icmp_ln42_fu_140_p2 == 1'd1)) | ((1'b1 == ap_block_state3_io) & (ap_enable_reg_pp0_iter1 == 1'b1)));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = (((fifo_0_vld_out == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (icmp_ln42_fu_140_p2 == 1'd1)) | ((1'b1 == ap_block_state3_io) & (ap_enable_reg_pp0_iter1 == 1'b1)));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter0 = ((fifo_0_vld_out == 1'b0) & (icmp_ln42_fu_140_p2 == 1'd1));
end

always @ (*) begin
    ap_block_state3_io = ((buf_r_WREADY == 1'b0) & (icmp_ln42_reg_168 == 1'd1));
end

assign ap_block_state3_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state8 = ((buf_r_BVALID == 1'b0) & (icmp_ln44_reg_182 == 1'd0));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign buf_offset1_fu_115_p4 = {{buf_offset[31:2]}};

assign empty_fu_125_p1 = buf_offset1_fu_115_p4;

assign fifo_0_ack_in = fifo_0_state[1'd1];

assign fifo_0_load_A = (fifo_0_state_cmp_full & ~fifo_0_sel_wr);

assign fifo_0_load_B = (fifo_0_state_cmp_full & fifo_0_sel_wr);

assign fifo_0_sel = fifo_0_sel_rd;

assign fifo_0_state_cmp_full = ((fifo_0_state != 2'd1) ? 1'b1 : 1'b0);

assign fifo_0_vld_in = fifo_TVALID;

assign fifo_0_vld_out = fifo_0_state[1'd0];

assign fifo_TREADY = fifo_0_state[1'd1];

assign i_fu_145_p2 = (i_0_reg_104 + 31'd1);

assign icmp_ln42_fu_140_p2 = (($signed(zext_ln42_fu_136_p1) < $signed(len)) ? 1'b1 : 1'b0);

assign icmp_ln44_fu_151_p2 = ((len == 32'd0) ? 1'b1 : 1'b0);

assign zext_ln42_fu_136_p1 = i_0_reg_104;

endmodule //a2_s2mm_data_copy
